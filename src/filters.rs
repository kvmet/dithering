use image::{DynamicImage, GrayImage, Luma, RgbImage, Rgb};
use png::{BitDepth, ColorType, Encoder};
use std::fs::File;
use std::io::{self, BufWriter};
use std::path::Path;
use crate::kernel_expander;


/// Convert sRGB value to linear light (gamma expansion)
fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear light to sRGB (gamma compression)
#[allow(dead_code)]
fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// A threshold kernel for ordered dithering
///
/// Each value represents a brightness threshold (0.0 to 1.0) for that position.
/// When processing an image, if a pixel's brightness exceeds the threshold at that
/// kernel position, it becomes white; otherwise black.
#[derive(Clone, Debug)]
pub struct ThresholdKernel {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f64>,
}

impl ThresholdKernel {
    /// Create a new threshold kernel with the given dimensions and values
    ///
    /// Values should be in the range [0.0, 1.0]
    pub fn new(width: usize, height: usize, values: Vec<f64>) -> Self {
        assert_eq!(width * height, values.len(), "Kernel dimensions don't match values length");
        Self { width, height, values }
    }

    /// Get the threshold value at position (x, y) in the kernel
    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.values[y * self.width + x]
    }

    /// Generate a kernel from a discrete seed
    ///
    /// Creates evenly distributed threshold values and shuffles them deterministically
    /// based on the seed. Same seed always produces the same kernel.
    /// Note: Similar seeds produce completely different patterns (high avalanche).
    pub fn from_seed(width: usize, height: usize, seed: u64) -> Self {
        let total = width * height;

        // Generate evenly distributed values
        let mut values: Vec<f64> = (0..total)
            .map(|i| (i + 1) as f64 / (total + 1) as f64)
            .collect();

        // Deterministic shuffle using a simple LCG (Linear Congruential Generator)
        let mut rng_state = seed;
        for i in (1..total).rev() {
            // LCG: next = (a * state + c) mod m
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            values.swap(i, j);
        }

        Self::new(width, height, values)
    }

    /// Generate a kernel from a continuous seed
    ///
    /// Similar to from_seed, but with low avalanche - similar seeds produce similar shuffle patterns.
    /// Still uses evenly distributed threshold values like from_seed.
    /// This allows for continuous optimization where nearby seeds produce similar dither patterns.
    pub fn from_seed_continuous(width: usize, height: usize, seed: f32) -> Self {
        let total = width * height;

        // Generate evenly distributed values (same as discrete version)
        let mut values: Vec<f64> = (0..total)
            .map(|i| (i + 1) as f64 / (total + 1) as f64)
            .collect();

        // Create a smooth shuffle using seed-controlled swap interpolation
        // Instead of discrete swaps, we use fractional positions
        for i in (1..total).rev() {
            // Generate a smooth pseudo-random value from seed and position
            // Using multiple sine waves for smooth variation
            let t1 = ((seed + i as f32 * 0.1) * 0.7).sin();
            let t2 = ((seed * 1.3 + i as f32 * 0.3) * 0.5).cos();
            let combined = (t1 + t2) * 0.5 + 0.5; // Normalize to 0.0-1.0

            // Map to a swap index
            let j = (combined * (i + 1) as f32) as usize % (i + 1);
            values.swap(i, j);
        }

        Self::new(width, height, values)
    }

    /// The example 3x3 kernel from the description (29%, 1%, 47%, etc.)
    /// Values are squared percentages normalized to 0-1 range
    pub fn example_3x3() -> Self {
        let values = vec![
            (29.0 * 29.0) / 10000.0, (1.0 * 1.0) / 10000.0, (47.0 * 47.0) / 10000.0,
            (41.0 * 41.0) / 10000.0, (37.0 * 37.0) / 10000.0, (1.0 * 1.0) / 10000.0,
            (23.0 * 23.0) / 10000.0, (41.0 * 41.0) / 10000.0, (29.0 * 29.0) / 10000.0,
        ];
        Self::new(3, 3, values)
    }

    /// Create an optimized 3x3 kernel with evenly distributed thresholds
    /// that are maximally different from their neighbors
    pub fn optimized_3x3() -> Self {
        // Values chosen to:
        // 1. Spread thresholds evenly (1/10, 2/10, ..., 9/10)
        // 2. Maximize difference between adjacent cells
        let values = vec![
            2.0/16.0, 9.0/16.0, 5.0/16.0, 14.0/16.0,
            11.0/16.0, 4.0/16.0, 16.0/16.0, 7.0/16.0,
            15.0/16.0, 8.0/16.0, 12.0/16.0, 3.0/16.0,
            6.0/16.0, 13.0/16.0, 1.0/16.0, 10.0/16.0,
        ];
        Self::new(4, 4, values)
    }

    /// Generate an optimized kernel using simulated annealing
    ///
    /// Creates a kernel with evenly distributed threshold values that are arranged
    /// to maximize differences between adjacent cells (including wrapping edges).
    /// Uses simulated annealing to find near-optimal arrangements.
    ///
    /// Supported sizes: 3x3, 4x4, 5x5
    ///
    /// # Arguments
    /// * `width` - Kernel width (must equal height - only square kernels supported)
    /// * `height` - Kernel height (must equal width - only square kernels supported)
    /// * `seed` - Random seed for optimization (can be numeric or alphanumeric string converted to u64)
    pub fn from_annealing(width: usize, height: usize, seed: u64) -> Self {
        assert_eq!(width, height, "Only square kernels are supported");
        let size = width;
        let total = size * size;

        // Determine iteration counts based on size
        let iterations = match size {
            2 => 100_000,
            3 => 1_000_000,
            4 => 4_500_000,
            5 => 15_000_000,
            _ => panic!("Unsupported kernel size: {}x{}", width, height),
        };

        // Use the optimized kernel_optimizer module with builder API
        let optimized_kernel = super::kernel_optimizer::OptimizerConfig::new(size, seed)
            .with_iterations(iterations)
            .optimize();

        // Convert from 1..N to 0..1 range
        let normalized: Vec<f64> = optimized_kernel.grid
            .iter()
            .map(|&v| v / (total + 1) as f64)
            .collect();

        Self::new(width, height, normalized)
    }

    /// Convert an alphanumeric seed string to a u64
    ///
    /// This allows for shorter, more memorable seeds like "abc123" instead of
    /// long numeric seeds like "18446744073709551615"
    pub fn seed_from_string(seed_str: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed_str.hash(&mut hasher);
        hasher.finish()
    }





    /// Standard 2x2 Bayer matrix
    pub fn bayer_2x2() -> Self {
        let values = vec![
            0.0, 0.5,
            0.75, 0.25,
        ];
        Self::new(2, 2, values)
    }

    /// Standard 4x4 Bayer matrix
    pub fn bayer_4x4() -> Self {
        let values = vec![
            0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
            12.0/16.0, 4.0/16.0, 14.0/16.0,  6.0/16.0,
            3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
            15.0/16.0, 7.0/16.0, 13.0/16.0,  5.0/16.0,
        ];
        Self::new(4, 4, values)
    }
}

/// Apply threshold kernel dithering with perceptual brightness (no normalization)
///
/// Uses gamma correction but does not normalize based on percentiles.
pub fn apply_threshold_kernel_perceptual(img: &DynamicImage, kernel: &ThresholdKernel, gamma: f32) -> GrayImage {
    apply_threshold_kernel_internal(img, kernel, true, false, gamma)
}

/// Apply threshold kernel dithering with percentile-based normalization and perceptual brightness
///
/// Combines both percentile normalization and gamma correction for maximum detail preservation.
pub fn apply_threshold_kernel_normalized_perceptual(img: &DynamicImage, kernel: &ThresholdKernel, gamma: f32) -> GrayImage {
    apply_threshold_kernel_internal(img, kernel, true, true, gamma)
}

/// Internal function to apply threshold kernel dithering
fn apply_threshold_kernel_internal(img: &DynamicImage, kernel: &ThresholdKernel, perceptual: bool, normalize: bool, gamma: f32) -> GrayImage {
    if normalize {
        apply_threshold_kernel_normalized_internal(img, kernel, perceptual, gamma)
    } else {
        apply_threshold_kernel_direct(img, kernel, perceptual, gamma)
    }
}

/// Apply threshold kernel directly without normalization
fn apply_threshold_kernel_direct(img: &DynamicImage, kernel: &ThresholdKernel, perceptual: bool, gamma: f32) -> GrayImage {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Create output image
    let mut output = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = gray.get_pixel(x, y);
            let mut brightness = pixel[0] as f32 / 255.0;

            // Apply gamma correction first (in encoded space)
            brightness = brightness.powf(gamma);

            if perceptual {
                brightness = srgb_to_linear(brightness);
            }

            // Tile the kernel across the image
            let kx = (x as usize) % kernel.width;
            let ky = (y as usize) % kernel.height;
            let threshold = kernel.get(kx, ky) as f32;

            // Apply threshold
            let output_value = if brightness > threshold { 255 } else { 0 };
            output.put_pixel(x, y, Luma([output_value]));
        }
    }

    output
}

/// Internal function to apply normalized threshold kernel dithering
fn apply_threshold_kernel_normalized_internal(img: &DynamicImage, kernel: &ThresholdKernel, perceptual: bool, gamma: f32) -> GrayImage {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Collect all brightness values
    let mut brightness_values: Vec<f32> = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = gray.get_pixel(x, y);
            let mut brightness = pixel[0] as f32 / 255.0;

            // Apply gamma correction first (in encoded space)
            brightness = brightness.powf(gamma);

            if perceptual {
                brightness = srgb_to_linear(brightness);
            }

            brightness_values.push(brightness);
        }
    }

    // Sort to compute percentiles
    brightness_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let total_pixels = brightness_values.len();

    // Create output image
    let mut output = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = gray.get_pixel(x, y);
            let mut brightness = pixel[0] as f32 / 255.0;

            // Apply gamma correction first (in encoded space)
            brightness = brightness.powf(gamma);

            if perceptual {
                brightness = srgb_to_linear(brightness);
            }

            // Find the percentile of this brightness value
            // Binary search to find where this value falls in the sorted array
            let percentile = brightness_values.partition_point(|&v| v < brightness) as f32 / total_pixels as f32;

            // Tile the kernel across the image
            let kx = (x as usize) % kernel.width;
            let ky = (y as usize) % kernel.height;
            let threshold = kernel.get(kx, ky) as f32;

            // Compare percentile to threshold
            let output_value = if percentile > threshold { 255 } else { 0 };
            output.put_pixel(x, y, Luma([output_value]));
        }
    }

    output
}

/// Measure the mean brightness of an image
pub fn measure_mean_brightness(img: &DynamicImage, perceptual: bool) -> f32 {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let mut sum = 0.0;
    let total_pixels = (width * height) as f32;

    for y in 0..height {
        for x in 0..width {
            let pixel = gray.get_pixel(x, y);
            let mut brightness = pixel[0] as f32 / 255.0;

            if perceptual {
                brightness = srgb_to_linear(brightness);
            }

            sum += brightness;
        }
    }

    sum / total_pixels
}

/// Compute automatic gamma correction to match input/output mean brightness
/// Only works for non-normalized mode (normalized mode doesn't need auto-gamma)
pub fn compute_auto_gamma(img: &DynamicImage, kernel: &ThresholdKernel, perceptual: bool, _normalized: bool) -> f32 {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Compute mean brightness of input (in linear space)
    let mut input_sum = 0.0;
    let total_pixels = (width * height) as f32;

    for y in 0..height {
        for x in 0..width {
            let pixel = gray.get_pixel(x, y);
            let mut brightness = pixel[0] as f32 / 255.0;

            if perceptual {
                brightness = srgb_to_linear(brightness);
            }

            input_sum += brightness;
        }
    }

    let input_mean = input_sum / total_pixels;

    // Try different gamma values and find one that matches input mean
    // Binary search for gamma that produces matching mean brightness
    let mut low = 0.1;
    let mut high = 3.0;
    let target = input_mean;

    for _ in 0..20 {
        let gamma = (low + high) / 2.0;
        let output_mean = estimate_output_mean(img, kernel, perceptual, gamma);

        if output_mean < target {
            // Output too dark, need lower gamma (brighter)
            high = gamma;
        } else {
            // Output too bright, need higher gamma (darker)
            low = gamma;
        }
    }

    (low + high) / 2.0
}

/// Estimate the mean brightness of dithered output for a given gamma
fn estimate_output_mean(img: &DynamicImage, kernel: &ThresholdKernel, perceptual: bool, gamma: f32) -> f32 {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let total_pixels = (width * height) as f32;
    let mut white_pixels = 0.0;

    for y in 0..height {
        for x in 0..width {
            let pixel = gray.get_pixel(x, y);
            let mut brightness = pixel[0] as f32 / 255.0;

            brightness = brightness.powf(gamma);

            if perceptual {
                brightness = srgb_to_linear(brightness);
            }

            let kx = (x as usize) % kernel.width;
            let ky = (y as usize) % kernel.height;
            let threshold = kernel.get(kx, ky) as f32;

            if brightness > threshold {
                white_pixels += 1.0;
            }
        }
    }

    white_pixels / total_pixels
}



/// Apply threshold kernel dithering to a color image with perceptual brightness (no normalization)
///
/// This applies the threshold kernel separately to each RGB channel, creating
/// an 8-color output (R/G/B each on or off = 2^3 = 8 colors).
pub fn apply_threshold_kernel_perceptual_color(img: &DynamicImage, kernel: &ThresholdKernel, gamma: f32) -> RgbImage {
    apply_threshold_kernel_color_internal(img, kernel, ColorMode::Normal, false, gamma)
}

/// Apply threshold kernel dithering to a color image with percentile-based normalization
///
/// This applies the threshold kernel separately to each RGB channel, creating
/// an 8-color output (R/G/B each on or off = 2^3 = 8 colors).
pub fn apply_threshold_kernel_normalized_perceptual_color(img: &DynamicImage, kernel: &ThresholdKernel, gamma: f32) -> RgbImage {
    apply_threshold_kernel_color_internal(img, kernel, ColorMode::Normal, true, gamma)
}

/// Apply threshold kernel dithering to a color image using dominant channel mode
///
/// For each pixel, determines which channel(s) have the strongest value.
/// Only the dominant channel(s) are kept, others set to 0.
/// Creates a more saturated look with primary colors: R, G, B, C, M, Y, K, W
///
/// `tie_threshold`: How close a channel must be to the max to be considered "tied" (0.0 to 1.0)
/// - 0.0 = only exact matches (most restrictive)
/// - 0.1 = within 10% of max can tie
/// - 0.5 = within 50% of max can tie (more permissive)
pub fn apply_threshold_kernel_perceptual_color_dominant(img: &DynamicImage, kernel: &ThresholdKernel, tie_threshold: f32, gamma: f32) -> RgbImage {
    apply_threshold_kernel_color_internal(img, kernel, ColorMode::Dominant(tie_threshold), false, gamma)
}

/// Apply threshold kernel dithering to a color image using dominant channel mode with normalization
///
/// For each pixel, determines which channel(s) have the strongest value.
/// Only the dominant channel(s) are kept, others set to 0.
/// Creates a more saturated look with primary colors: R, G, B, C, M, Y, K, W
///
/// `tie_threshold`: How close a channel must be to the max to be considered "tied" (0.0 to 1.0)
/// - 0.0 = only exact matches (most restrictive)
/// - 0.1 = within 10% of max can tie
/// - 0.5 = within 50% of max can tie (more permissive)
pub fn apply_threshold_kernel_normalized_perceptual_color_dominant(img: &DynamicImage, kernel: &ThresholdKernel, tie_threshold: f32, gamma: f32) -> RgbImage {
    apply_threshold_kernel_color_internal(img, kernel, ColorMode::Dominant(tie_threshold), true, gamma)
}

/// Apply threshold kernel dithering to a color image using exclusive mode
///
/// Only allows pure single channels (R, G, B) or white (all three).
/// Picks the single strongest channel, unless all three are similar (then white).
/// Creates only 4 colors: red, green, blue, black, white
pub fn apply_threshold_kernel_perceptual_color_exclusive(img: &DynamicImage, kernel: &ThresholdKernel, gamma: f32) -> RgbImage {
    apply_threshold_kernel_color_internal(img, kernel, ColorMode::Exclusive, false, gamma)
}

/// Apply threshold kernel dithering to a color image using exclusive mode with normalization
///
/// Only allows pure single channels (R, G, B) or white (all three).
/// Picks the single strongest channel, unless all three are similar (then white).
/// Creates only 4 colors: red, green, blue, black, white
pub fn apply_threshold_kernel_normalized_perceptual_color_exclusive(img: &DynamicImage, kernel: &ThresholdKernel, gamma: f32) -> RgbImage {
    apply_threshold_kernel_color_internal(img, kernel, ColorMode::Exclusive, true, gamma)
}

/// Apply threshold kernel dithering to a color image using CMYK mode
///
/// Uses expanded kernels - each channel (C, M, Y, K) gets a different kernel derived from the root.
/// This minimizes spatial overlap between channels, reducing artifacts.
/// Creates only 5 colors: cyan, magenta, yellow, black, white
pub fn apply_threshold_kernel_perceptual_color_cmyk(img: &DynamicImage, kernel: &ThresholdKernel, gamma: f32) -> RgbImage {
    apply_threshold_kernel_cmyk_direct(img, kernel, gamma)
}

/// Apply threshold kernel dithering to a color image using CMYK mode with normalization
///
/// Uses expanded kernels - each channel (C, M, Y, K) gets a different kernel derived from the root.
/// This minimizes spatial overlap between channels, reducing artifacts.
/// Creates only 5 colors: cyan, magenta, yellow, black, white
pub fn apply_threshold_kernel_normalized_perceptual_color_cmyk(img: &DynamicImage, kernel: &ThresholdKernel, gamma: f32) -> RgbImage {
    apply_threshold_kernel_cmyk_normalized(img, kernel, gamma)
}

enum ColorMode {
    Normal,
    Dominant(f32), // tie_threshold
    Exclusive,
    Cmyk,
}

fn apply_threshold_kernel_color_internal(img: &DynamicImage, kernel: &ThresholdKernel, mode: ColorMode, normalize: bool, gamma: f32) -> RgbImage {
    if normalize {
        apply_threshold_kernel_color_normalized_internal(img, kernel, mode, gamma)
    } else {
        apply_threshold_kernel_color_direct(img, kernel, mode, gamma)
    }
}

/// Apply threshold kernel to color image directly without normalization
fn apply_threshold_kernel_color_direct(img: &DynamicImage, kernel: &ThresholdKernel, mode: ColorMode, gamma: f32) -> RgbImage {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut output = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);

            // Apply gamma correction first (in encoded space)
            let mut r = (pixel[0] as f32 / 255.0).powf(gamma);
            let mut g = (pixel[1] as f32 / 255.0).powf(gamma);
            let mut b = (pixel[2] as f32 / 255.0).powf(gamma);

            r = srgb_to_linear(r);
            g = srgb_to_linear(g);
            b = srgb_to_linear(b);

            // Tile the kernel across the image
            let kx = (x as usize) % kernel.width;
            let ky = (y as usize) % kernel.height;
            let threshold = kernel.get(kx, ky) as f32;

            let mut r_out = if r > threshold { 255 } else { 0 };
            let mut g_out = if g > threshold { 255 } else { 0 };
            let mut b_out = if b > threshold { 255 } else { 0 };

            // Apply color mode logic
            match mode {
                ColorMode::Normal => {
                    // All channels dither independently - do nothing
                }
                ColorMode::Dominant(tie_thresh) => {
                    if r_out > 0 || g_out > 0 || b_out > 0 {
                        let max_val = r.max(g).max(b);
                        let threshold_value = max_val * tie_thresh;
                        if r < max_val - threshold_value { r_out = 0; }
                        if g < max_val - threshold_value { g_out = 0; }
                        if b < max_val - threshold_value { b_out = 0; }
                    }
                }
                ColorMode::Exclusive => {
                    if r_out > 0 || g_out > 0 || b_out > 0 {
                        if r_out > 0 && g_out > 0 && b_out > 0 {
                            // Keep white
                        } else {
                            let max_val = r.max(g).max(b);
                            if r == max_val {
                                g_out = 0;
                                b_out = 0;
                            } else if g == max_val {
                                r_out = 0;
                                b_out = 0;
                            } else {
                                r_out = 0;
                                g_out = 0;
                            }
                        }
                    }
                }
                ColorMode::Cmyk => {
                    if r_out > 0 || g_out > 0 || b_out > 0 {
                        if r_out > 0 && g_out > 0 && b_out > 0 {
                            // All on = white, keep it
                        } else if r_out == 0 && g_out == 0 && b_out == 0 {
                            // All off = black, keep it
                        } else {
                            // Pick the strongest channel and invert it to get CMY
                            let max_val = r.max(g).max(b);
                            if r == max_val {
                                // Red is strongest, turn it off -> Cyan (0, 255, 255)
                                r_out = 0;
                                g_out = 255;
                                b_out = 255;
                            } else if g == max_val {
                                // Green is strongest, turn it off -> Magenta (255, 0, 255)
                                r_out = 255;
                                g_out = 0;
                                b_out = 255;
                            } else {
                                // Blue is strongest, turn it off -> Yellow (255, 255, 0)
                                r_out = 255;
                                g_out = 255;
                                b_out = 0;
                            }
                        }
                    }
                }
            }

            output.put_pixel(x, y, Rgb([r_out, g_out, b_out]));
        }
    }

    output
}

fn apply_threshold_kernel_color_normalized_internal(img: &DynamicImage, kernel: &ThresholdKernel, mode: ColorMode, gamma: f32) -> RgbImage {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Collect brightness values for each channel separately
    let mut red_values: Vec<f32> = Vec::with_capacity((width * height) as usize);
    let mut green_values: Vec<f32> = Vec::with_capacity((width * height) as usize);
    let mut blue_values: Vec<f32> = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);

            // Apply gamma correction first (in encoded space)
            let mut r = (pixel[0] as f32 / 255.0).powf(gamma);
            let mut g = (pixel[1] as f32 / 255.0).powf(gamma);
            let mut b = (pixel[2] as f32 / 255.0).powf(gamma);

            r = srgb_to_linear(r);
            g = srgb_to_linear(g);
            b = srgb_to_linear(b);

            red_values.push(r);
            green_values.push(g);
            blue_values.push(b);
        }
    }

    // Sort each channel to compute percentiles
    red_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    green_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    blue_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let total_pixels = (width * height) as usize;

    // Create output image
    let mut output = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);

            // Apply gamma correction first (in encoded space)
            let mut r = (pixel[0] as f32 / 255.0).powf(gamma);
            let mut g = (pixel[1] as f32 / 255.0).powf(gamma);
            let mut b = (pixel[2] as f32 / 255.0).powf(gamma);

            r = srgb_to_linear(r);
            g = srgb_to_linear(g);
            b = srgb_to_linear(b);

            // Find percentile for each channel
            let r_percentile = red_values.partition_point(|&v| v < r) as f32 / total_pixels as f32;
            let g_percentile = green_values.partition_point(|&v| v < g) as f32 / total_pixels as f32;
            let b_percentile = blue_values.partition_point(|&v| v < b) as f32 / total_pixels as f32;

            // Tile the kernel across the image
            let kx = (x as usize) % kernel.width;
            let ky = (y as usize) % kernel.height;
            let threshold = kernel.get(kx, ky) as f32;

            let mut r_out = if r_percentile > threshold { 255 } else { 0 };
            let mut g_out = if g_percentile > threshold { 255 } else { 0 };
            let mut b_out = if b_percentile > threshold { 255 } else { 0 };

            // Apply color mode logic
            match mode {
                ColorMode::Normal => {
                    // All channels dither independently - do nothing
                }
                ColorMode::Dominant(tie_thresh) => {
                    if r_out > 0 || g_out > 0 || b_out > 0 {
                        let max_val = r.max(g).max(b);
                        let threshold_value = max_val * tie_thresh;
                        if r < max_val - threshold_value { r_out = 0; }
                        if g < max_val - threshold_value { g_out = 0; }
                        if b < max_val - threshold_value { b_out = 0; }
                    }
                }
                ColorMode::Exclusive => {
                    if r_out > 0 || g_out > 0 || b_out > 0 {
                        if r_out > 0 && g_out > 0 && b_out > 0 {
                            // Keep white
                        } else {
                            let max_val = r.max(g).max(b);
                            if r == max_val {
                                g_out = 0;
                                b_out = 0;
                            } else if g == max_val {
                                r_out = 0;
                                b_out = 0;
                            } else {
                                r_out = 0;
                                g_out = 0;
                            }
                        }
                    }
                }
                ColorMode::Cmyk => {
                    if r_out > 0 || g_out > 0 || b_out > 0 {
                        if r_out > 0 && g_out > 0 && b_out > 0 {
                            // All on = white, keep it
                        } else if r_out == 0 && g_out == 0 && b_out == 0 {
                            // All off = black, keep it
                        } else {
                            // Pick the strongest channel and invert it to get CMY
                            let max_val = r.max(g).max(b);
                            if r == max_val {
                                // Red is strongest, turn it off -> Cyan (0, 255, 255)
                                r_out = 0;
                                g_out = 255;
                                b_out = 255;
                            } else if g == max_val {
                                // Green is strongest, turn it off -> Magenta (255, 0, 255)
                                r_out = 255;
                                g_out = 0;
                                b_out = 255;
                            } else {
                                // Blue is strongest, turn it off -> Yellow (255, 255, 0)
                                r_out = 255;
                                g_out = 255;
                                b_out = 0;
                            }
                        }
                    }
                }
            }

            output.put_pixel(x, y, Rgb([r_out, g_out, b_out]));
        }
    }

    output
}

/// Apply CMYK dithering using expanded kernels (direct, no normalization)
fn apply_threshold_kernel_cmyk_direct(img: &DynamicImage, kernel: &ThresholdKernel, gamma: f32) -> RgbImage {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Expand the root kernel into CMYK kernels
    let cmyk_kernels = kernel_expander::expand_kernel_cmyk(&kernel.values, kernel.width);

    let mut output = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);

            // Apply gamma correction first (in encoded space)
            let mut r = (pixel[0] as f32 / 255.0).powf(gamma);
            let mut g = (pixel[1] as f32 / 255.0).powf(gamma);
            let mut b = (pixel[2] as f32 / 255.0).powf(gamma);

            r = srgb_to_linear(r);
            g = srgb_to_linear(g);
            b = srgb_to_linear(b);

            // Tile the kernel across the image
            let kx = (x as usize) % kernel.width;
            let ky = (y as usize) % kernel.height;
            let kid = ky * kernel.width + kx;

            // Get thresholds from each channel's kernel
            let threshold_c = cmyk_kernels.cyan[kid] as f32;
            let threshold_m = cmyk_kernels.magenta[kid] as f32;
            let threshold_y = cmyk_kernels.yellow[kid] as f32;
            let threshold_k = cmyk_kernels.black[kid] as f32;

            // Determine which channel wins based on color values
            // CMY is inverted from RGB, and K is for when all are low
            let cyan_strength = 1.0 - r;      // lack of red
            let magenta_strength = 1.0 - g;   // lack of green
            let yellow_strength = 1.0 - b;    // lack of blue
            let black_strength = 1.0 - r.max(g).max(b);  // lack of brightness

            // Check which channels activate at this position
            let c_on = cyan_strength > threshold_c;
            let m_on = magenta_strength > threshold_m;
            let y_on = yellow_strength > threshold_y;
            let k_on = black_strength > threshold_k;

            let (r_out, g_out, b_out) = if k_on {
                // Black wins
                (0, 0, 0)
            } else if !c_on && !m_on && !y_on {
                // Nothing activated = white
                (255, 255, 255)
            } else {
                // Determine winner among CMY channels
                let max_strength = cyan_strength.max(magenta_strength).max(yellow_strength);

                if cyan_strength == max_strength && c_on {
                    (0, 255, 255)  // Cyan
                } else if magenta_strength == max_strength && m_on {
                    (255, 0, 255)  // Magenta
                } else if yellow_strength == max_strength && y_on {
                    (255, 255, 0)  // Yellow
                } else {
                    // Fallback to white if nothing wins
                    (255, 255, 255)
                }
            };

            output.put_pixel(x, y, Rgb([r_out, g_out, b_out]));
        }
    }

    output
}

/// Apply CMYK dithering using expanded kernels (with normalization)
fn apply_threshold_kernel_cmyk_normalized(img: &DynamicImage, kernel: &ThresholdKernel, gamma: f32) -> RgbImage {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Expand the root kernel into CMYK kernels
    let cmyk_kernels = kernel_expander::expand_kernel_cmyk(&kernel.values, kernel.width);

    // Collect CMY and K values for normalization
    let mut cyan_values: Vec<f32> = Vec::with_capacity((width * height) as usize);
    let mut magenta_values: Vec<f32> = Vec::with_capacity((width * height) as usize);
    let mut yellow_values: Vec<f32> = Vec::with_capacity((width * height) as usize);
    let mut black_values: Vec<f32> = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);

            let mut r = (pixel[0] as f32 / 255.0).powf(gamma);
            let mut g = (pixel[1] as f32 / 255.0).powf(gamma);
            let mut b = (pixel[2] as f32 / 255.0).powf(gamma);

            r = srgb_to_linear(r);
            g = srgb_to_linear(g);
            b = srgb_to_linear(b);

            cyan_values.push(1.0 - r);
            magenta_values.push(1.0 - g);
            yellow_values.push(1.0 - b);
            black_values.push(1.0 - r.max(g).max(b));
        }
    }

    // Sort for percentile calculation
    cyan_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    magenta_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    yellow_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    black_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let total_pixels = (width * height) as usize;

    let mut output = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);

            let mut r = (pixel[0] as f32 / 255.0).powf(gamma);
            let mut g = (pixel[1] as f32 / 255.0).powf(gamma);
            let mut b = (pixel[2] as f32 / 255.0).powf(gamma);

            r = srgb_to_linear(r);
            g = srgb_to_linear(g);
            b = srgb_to_linear(b);

            let cyan_strength = 1.0 - r;
            let magenta_strength = 1.0 - g;
            let yellow_strength = 1.0 - b;
            let black_strength = 1.0 - r.max(g).max(b);

            // Calculate percentiles
            let c_percentile = cyan_values.partition_point(|&v| v < cyan_strength) as f32 / total_pixels as f32;
            let m_percentile = magenta_values.partition_point(|&v| v < magenta_strength) as f32 / total_pixels as f32;
            let y_percentile = yellow_values.partition_point(|&v| v < yellow_strength) as f32 / total_pixels as f32;
            let k_percentile = black_values.partition_point(|&v| v < black_strength) as f32 / total_pixels as f32;

            // Tile the kernel across the image
            let kx = (x as usize) % kernel.width;
            let ky = (y as usize) % kernel.height;
            let kid = ky * kernel.width + kx;

            // Get thresholds from each channel's kernel
            let threshold_c = cmyk_kernels.cyan[kid] as f32;
            let threshold_m = cmyk_kernels.magenta[kid] as f32;
            let threshold_y = cmyk_kernels.yellow[kid] as f32;
            let threshold_k = cmyk_kernels.black[kid] as f32;

            // Check which channels activate
            let c_on = c_percentile > threshold_c;
            let m_on = m_percentile > threshold_m;
            let y_on = y_percentile > threshold_y;
            let k_on = k_percentile > threshold_k;

            let (r_out, g_out, b_out) = if k_on {
                // Black wins
                (0, 0, 0)
            } else if !c_on && !m_on && !y_on {
                // Nothing activated = white
                (255, 255, 255)
            } else {
                // Determine winner among CMY channels
                let max_strength = cyan_strength.max(magenta_strength).max(yellow_strength);

                if cyan_strength == max_strength && c_on {
                    (0, 255, 255)  // Cyan
                } else if magenta_strength == max_strength && m_on {
                    (255, 0, 255)  // Magenta
                } else if yellow_strength == max_strength && y_on {
                    (255, 255, 0)  // Yellow
                } else {
                    // Fallback to white if nothing wins
                    (255, 255, 255)
                }
            };

            output.put_pixel(x, y, Rgb([r_out, g_out, b_out]));
        }
    }

    output
}

/// Save a color image as 8-bit PNG
///
/// Standard RGB PNG encoding.
pub fn save_as_color_png<P: AsRef<Path>>(img: &RgbImage, path: P) -> io::Result<()> {
    let (width, height) = img.dimensions();
    let file = File::create(path)?;
    let buf_writer = BufWriter::new(file);

    let mut encoder = Encoder::new(buf_writer, width, height);
    encoder.set_color(ColorType::Rgb);
    encoder.set_depth(BitDepth::Eight);

    let mut writer = encoder.write_header()?;

    // RGB data is already in the right format
    writer.write_image_data(img.as_raw())?;

    Ok(())
}

/// Save a monochrome image as 1-bit PNG
///
/// This uses the png crate directly to encode a true 1-bit-per-pixel PNG,
/// which is much more space-efficient than the default 8-bit grayscale PNG.
/// A 400x400 image will be ~2-5KB instead of ~32KB.
pub fn save_as_1bit_png<P: AsRef<Path>>(img: &GrayImage, path: P) -> io::Result<()> {
    let (width, height) = img.dimensions();
    let file = File::create(path)?;
    let buf_writer = BufWriter::new(file);

    let mut encoder = Encoder::new(buf_writer, width, height);
    encoder.set_color(ColorType::Grayscale);
    encoder.set_depth(BitDepth::One);

    let mut writer = encoder.write_header()?;

    // Pack pixels into bits
    // PNG stores rows, each row padded to byte boundary
    let bytes_per_row = (width + 7) / 8; // Round up to nearest byte
    let total_bytes = bytes_per_row * height;
    let mut data = vec![0u8; total_bytes as usize];

    for y in 0..height {
        let row_offset = (y * bytes_per_row) as usize;

        for x in 0..width {
            let pixel = img.get_pixel(x, y)[0];
            // PNG 1-bit: 0 = black, 1 = white
            // Our images have 0 = black, 255 = white
            if pixel >= 128 {
                let byte_idx = row_offset + (x / 8) as usize;
                let bit_idx = 7 - (x % 8); // MSB first
                data[byte_idx] |= 1 << bit_idx;
            }
        }
    }

    writer.write_image_data(&data)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        let kernel = ThresholdKernel::new(3, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        assert_eq!(kernel.width, 3);
        assert_eq!(kernel.height, 3);
        assert_eq!(kernel.get(1, 1), 0.5);
    }

    #[test]
    fn test_example_kernel() {
        let kernel = ThresholdKernel::example_3x3();
        assert_eq!(kernel.width, 3);
        assert_eq!(kernel.height, 3);
    }
}
