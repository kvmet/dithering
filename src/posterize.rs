use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use crate::filters::ThresholdKernel;

/// Posterize an image with CMYK colors using dithering on the K channel
///
/// Algorithm:
/// 1. Blur the input image to create larger color blocks
/// 2. Convert to monochrome and apply threshold dithering (K channel)
/// 3. For white pixels in the dithered output, sample the blurred image
///    and replace with the dominant CMYK color (excluding pure black/K)
///
/// This creates a posterized effect with dithered black areas and solid
/// color blocks (C, M, Y, R, G, B, W) in the non-black regions.
pub fn posterize_cmyk_dithered(
    img: &DynamicImage,
    kernel: &ThresholdKernel,
    blur_radius: f32,
    gamma: f32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    // Step 1: Blur the image to create larger color blocks
    let blurred = if blur_radius > 0.0 {
        image::imageops::blur(img, blur_radius)
    } else {
        img.to_rgba8()
    };

    // Step 2: Create monochrome dithered version (K channel)
    let dithered_mono = dither_monochrome(img, kernel, gamma);

    // Step 3: Create output by combining dithered K with posterized colors
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let dithered_pixel = dithered_mono.get_pixel(x, y);

            // If the dithered pixel is white (not K), replace with posterized color
            if dithered_pixel[0] > 128 {
                // Sample the blurred image to get posterized color
                let blurred_pixel = blurred.get_pixel(x, y);
                let cmyk_color = quantize_to_cmyk_no_k(blurred_pixel);
                output.put_pixel(x, y, cmyk_color);
            } else {
                // Black pixel from dithering - keep it black (K)
                output.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
    }

    output
}

/// Dither an image to monochrome using perceptual brightness
fn dither_monochrome(
    img: &DynamicImage,
    kernel: &ThresholdKernel,
    gamma: f32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    let rgb_img = img.to_rgb8();

    // Convert to perceptual grayscale with gamma correction
    let mut gray = vec![0.0f64; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_img.get_pixel(x, y);
            let r = (pixel[0] as f64 / 255.0).powf(2.2);
            let g = (pixel[1] as f64 / 255.0).powf(2.2);
            let b = (pixel[2] as f64 / 255.0).powf(2.2);

            // Perceptual luminance
            let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;

            // Apply gamma correction
            let adjusted = luminance.powf(gamma as f64);

            gray[(y * width + x) as usize] = adjusted;
        }
    }

    // Apply error diffusion dithering
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let old_val = gray[idx];

            // Get threshold from kernel
            let threshold = kernel.get(
                (x as usize) % kernel.width,
                (y as usize) % kernel.height,
            );

            // Threshold the pixel
            let new_val = if old_val > threshold { 1.0 } else { 0.0 };
            let error = old_val - new_val;

            // Set output pixel
            let out_color = if new_val > 0.5 { 255 } else { 0 };
            output.put_pixel(x, y, Rgb([out_color, out_color, out_color]));

            // Distribute error using Floyd-Steinberg weights
            // (We're using ordered dithering with the kernel, but still distribute error
            // to help reduce pattern artifacts)
            if x + 1 < width {
                gray[(y * width + x + 1) as usize] += error * 7.0 / 16.0;
            }
            if y + 1 < height {
                if x > 0 {
                    gray[((y + 1) * width + x - 1) as usize] += error * 3.0 / 16.0;
                }
                gray[((y + 1) * width + x) as usize] += error * 5.0 / 16.0;
                if x + 1 < width {
                    gray[((y + 1) * width + x + 1) as usize] += error * 1.0 / 16.0;
                }
            }
        }
    }

    output
}

/// Quantize a color to CMYK palette, excluding pure black (K)
///
/// Returns one of: C, M, Y, R, G, B, or W (white)
fn quantize_to_cmyk_no_k(pixel: &image::Rgba<u8>) -> Rgb<u8> {
    let r = pixel[0] as f32 / 255.0;
    let g = pixel[1] as f32 / 255.0;
    let b = pixel[2] as f32 / 255.0;

    // Convert to CMYK-like representation
    // Find which channels are dominant
    let max_component = r.max(g).max(b);

    // If very dark, we'd return K, but we exclude K, so return closest color
    if max_component < 0.2 {
        // Very dark - pick the least dark channel
        if r >= g && r >= b {
            return Rgb([255, 0, 0]); // R
        } else if g >= b {
            return Rgb([0, 255, 0]); // G
        } else {
            return Rgb([0, 0, 255]); // B
        }
    }

    // If very bright, return white
    if max_component > 0.8 && (r > 0.6 && g > 0.6 && b > 0.6) {
        return Rgb([255, 255, 255]); // W
    }

    // Determine dominant color(s)
    let threshold = max_component * 0.6; // Threshold for "on" channels

    let r_on = r > threshold;
    let g_on = g > threshold;
    let b_on = b > threshold;

    // Map to CMYK colors (excluding K)
    match (r_on, g_on, b_on) {
        (true, true, true) => Rgb([255, 255, 255]),   // W (white)
        (true, true, false) => Rgb([255, 255, 0]),     // Y (yellow) = R+G
        (true, false, true) => Rgb([255, 0, 255]),     // M (magenta) = R+B
        (false, true, true) => Rgb([0, 255, 255]),     // C (cyan) = G+B
        (true, false, false) => Rgb([255, 0, 0]),      // R (red)
        (false, true, false) => Rgb([0, 255, 0]),      // G (green)
        (false, false, true) => Rgb([0, 0, 255]),      // B (blue)
        (false, false, false) => {
            // All off - shouldn't happen with our checks above, but default to white
            Rgb([255, 255, 255])
        }
    }
}

/// Posterize with adjustable color levels per channel
///
/// A more traditional posterize that quantizes each RGB channel independently
#[allow(dead_code)]
pub fn posterize_rgb(
    img: &DynamicImage,
    levels: u8,
    blur_radius: f32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    // Blur first if requested
    let source = if blur_radius > 0.0 {
        image::imageops::blur(img, blur_radius)
    } else {
        img.to_rgba8()
    };

    let mut output = ImageBuffer::new(width, height);
    let step = 255.0 / (levels - 1) as f32;

    for y in 0..height {
        for x in 0..width {
            let pixel = source.get_pixel(x, y);

            let r = ((pixel[0] as f32 / step).round() * step) as u8;
            let g = ((pixel[1] as f32 / step).round() * step) as u8;
            let b = ((pixel[2] as f32 / step).round() * step) as u8;

            output.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    output
}
