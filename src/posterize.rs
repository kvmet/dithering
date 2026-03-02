use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use crate::filters::ThresholdKernel;

/// Posterize an image with CMY spread (no dithering - just posterization)
///
/// Algorithm:
/// 1. Extract C, M, Y channels as separate boolean masks
/// 2. Spread each channel independently by the radius
/// 3. Recombine CMY back to RGB
///
/// This creates a posterized effect with bloomed color areas.
/// Dithering happens separately when this is combined with the K channel.
pub fn posterize_cmy_spread(
    img: &DynamicImage,
    spread_radius: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    // Step 1: Extract C, M, Y channels as separate boolean masks
    let (c_mask, m_mask, y_mask) = extract_cmy_channels(img);

    // Step 2: Spread each channel independently
    let c_spread = spread_channel(&c_mask, spread_radius);
    let m_spread = spread_channel(&m_mask, spread_radius);
    let y_spread = spread_channel(&y_mask, spread_radius);

    // Step 3: Recombine CMY back to RGB
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let c = c_spread.get_pixel(x, y)[0] > 128;
            let m = m_spread.get_pixel(x, y)[0] > 128;
            let y_cmy = y_spread.get_pixel(x, y)[0] > 128;

            // Convert CMY back to RGB
            let r = if c { 0 } else { 255 };
            let g = if m { 0 } else { 255 };
            let b = if y_cmy { 0 } else { 255 };

            output.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    output
}

/// Extract C, M, Y channels as separate boolean masks
/// Returns (C_mask, M_mask, Y_mask) where 255 = channel is on, 0 = channel is off
fn extract_cmy_channels(img: &DynamicImage) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, ImageBuffer<Rgb<u8>, Vec<u8>>, ImageBuffer<Rgb<u8>, Vec<u8>>) {
    let (width, height) = img.dimensions();
    let mut c_mask = ImageBuffer::new(width, height);
    let mut m_mask = ImageBuffer::new(width, height);
    let mut y_mask = ImageBuffer::new(width, height);

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

            // Remove K component from CMY to get pure color components
            let c_adj = if k < 1.0 { (c - k) / (1.0 - k) } else { 0.0 };
            let m_adj = if k < 1.0 { (m - k) / (1.0 - k) } else { 0.0 };
            let y_adj = if k < 1.0 { (y_cmy - k) / (1.0 - k) } else { 0.0 };

            // Threshold each channel independently (>0.5 = on)
            let c_on = if c_adj > 0.5 { 255 } else { 0 };
            let m_on = if m_adj > 0.5 { 255 } else { 0 };
            let y_on = if y_adj > 0.5 { 255 } else { 0 };

            c_mask.put_pixel(x, y, Rgb([c_on, c_on, c_on]));
            m_mask.put_pixel(x, y, Rgb([m_on, m_on, m_on]));
            y_mask.put_pixel(x, y, Rgb([y_on, y_on, y_on]));
        }
    }

    (c_mask, m_mask, y_mask)
}

/// Spread a single channel mask using morphological dilation
/// Input: grayscale image where 255 = channel on, 0 = channel off
fn spread_channel(mask: &ImageBuffer<Rgb<u8>, Vec<u8>>, radius: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    if radius == 0 {
        return mask.clone();
    }

    let (width, height) = mask.dimensions();
    let mut current = mask.clone();

    // Dilate iteratively, one pixel at a time, for 'radius' iterations
    for _ in 0..radius {
        let mut next = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let original = current.get_pixel(x, y)[0];

                // If already on (255), keep it
                if original > 128 {
                    next.put_pixel(x, y, Rgb([255, 255, 255]));
                    continue;
                }

                // Check immediate neighbors (4-connected)
                let mut is_on = false;

                // Check up, down, left, right
                let neighbors = [
                    (x as i32, y as i32 - 1), // up
                    (x as i32, y as i32 + 1), // down
                    (x as i32 - 1, y as i32), // left
                    (x as i32 + 1, y as i32), // right
                ];

                for (nx, ny) in neighbors {
                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let neighbor = current.get_pixel(nx as u32, ny as u32)[0];

                        // If any neighbor is on, turn this pixel on
                        if neighbor > 128 {
                            is_on = true;
                            break;
                        }
                    }
                }

                let value = if is_on { 255 } else { 0 };
                next.put_pixel(x, y, Rgb([value, value, value]));
            }
        }

        current = next;
    }

    current
}

/// Combine posterized CMY with dithered K channel
///
/// Where the dithered K is black, use K; where it's white, use the posterized CMY color
pub fn combine_cmy_with_dithered_k(
    posterized_cmy: &ImageBuffer<Rgb<u8>, Vec<u8>>,
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

    // Apply error diffusion dithering and combine with posterized CMY
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

            // If dithered to black, use K; otherwise use posterized CMY
            if new_val < 0.5 {
                output.put_pixel(x, y, Rgb([0, 0, 0]));
            } else {
                let cmy_color = posterized_cmy.get_pixel(x, y);
                output.put_pixel(x, y, *cmy_color);
            }

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
