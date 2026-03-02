use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use crate::filters::ThresholdKernel;

/// Posterize an image with CMY spread and dithered K channel
///
/// Algorithm:
/// 1. Quantize all pixels to CMYKW palette
/// 2. Spread/bloom the CMY channels by a radius
/// 3. Dither the image to monochrome (K channel)
/// 4. For white pixels in dithered output, replace with the spread CMY color
///
/// This creates a screen-printing/halftone effect with bloomed color areas
/// and dithered black details.
pub fn posterize_cmy_spread(
    img: &DynamicImage,
    kernel: &ThresholdKernel,
    spread_radius: u32,
    gamma: f32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    // Step 1: Quantize to CMYKW
    let quantized = quantize_to_cmykw(img);

    // Step 2: Spread/bloom the CMY channels
    let spread_cmy = spread_cmy_channels(&quantized, spread_radius);

    // Step 3: Dither to monochrome (K channel)
    let dithered_k = dither_monochrome(img, kernel, gamma);

    // Step 4: Combine - use spread CMY where dithered is white, K where black
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let k_pixel = dithered_k.get_pixel(x, y);

            if k_pixel[0] > 128 {
                // White in dithered = use spread CMY color
                let cmy_color = spread_cmy.get_pixel(x, y);
                output.put_pixel(x, y, *cmy_color);
            } else {
                // Black in dithered = use K
                output.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
    }

    output
}

/// Quantize image to CMYKW palette
fn quantize_to_cmykw(img: &DynamicImage) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            // Convert to CMY
            let c = 1.0 - r;
            let m = 1.0 - g;
            let y_cmy = 1.0 - b;

            // Find K (black) component
            let k = c.min(m).min(y_cmy);

            // If mostly black, return K
            if k > 0.7 {
                output.put_pixel(x, y, Rgb([0, 0, 0]));
                continue;
            }

            // If very light, return W
            if k < 0.3 && c < 0.3 && m < 0.3 && y_cmy < 0.3 {
                output.put_pixel(x, y, Rgb([255, 255, 255]));
                continue;
            }

            // Remove K component from CMY
            let c_adj = if k < 1.0 { (c - k) / (1.0 - k) } else { 0.0 };
            let m_adj = if k < 1.0 { (m - k) / (1.0 - k) } else { 0.0 };
            let y_adj = if k < 1.0 { (y_cmy - k) / (1.0 - k) } else { 0.0 };

            // Determine which CMY channels are "on" (threshold at 0.5)
            let c_on = c_adj > 0.5;
            let m_on = m_adj > 0.5;
            let y_on = y_adj > 0.5;

            // Map back to RGB
            let color = match (c_on, m_on, y_on) {
                (true, true, true) => Rgb([0, 0, 0]),       // K (shouldn't happen but fallback)
                (true, true, false) => Rgb([0, 0, 255]),    // C+M = Blue
                (true, false, true) => Rgb([0, 255, 0]),    // C+Y = Green
                (false, true, true) => Rgb([255, 0, 0]),    // M+Y = Red
                (true, false, false) => Rgb([0, 255, 255]), // C = Cyan
                (false, true, false) => Rgb([255, 0, 255]), // M = Magenta
                (false, false, true) => Rgb([255, 255, 0]), // Y = Yellow
                (false, false, false) => Rgb([255, 255, 255]), // White
            };

            output.put_pixel(x, y, color);
        }
    }

    output
}

/// Spread/bloom the CMY channels (expand colored areas)
fn spread_cmy_channels(img: &ImageBuffer<Rgb<u8>, Vec<u8>>, radius: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    if radius == 0 {
        return img.clone();
    }

    let (width, height) = img.dimensions();
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let original = img.get_pixel(x, y);

            // If already colored (not K or W), keep it
            if *original != Rgb([0, 0, 0]) && *original != Rgb([255, 255, 255]) {
                output.put_pixel(x, y, *original);
                continue;
            }

            // Check neighborhood for CMY colors
            let mut found_color = Rgb([255, 255, 255]); // Default to white
            let mut found = false;

            'outer: for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;

                    let neighbor = img.get_pixel(nx, ny);

                    // If we find a CMY color (not K or W), use it
                    if *neighbor != Rgb([0, 0, 0]) && *neighbor != Rgb([255, 255, 255]) {
                        found_color = *neighbor;
                        found = true;
                        break 'outer; // Take first color found
                    }
                }
            }

            if found {
                output.put_pixel(x, y, found_color);
            } else {
                output.put_pixel(x, y, *original);
            }
        }
    }

    output
}

/// Dither image to monochrome using perceptual brightness
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
