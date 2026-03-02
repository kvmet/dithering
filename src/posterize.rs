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
    spread_offset: i32,
    spread_angle: f32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    // Step 1: Extract C, M, Y channels as separate boolean masks
    let (c_mask, m_mask, y_mask) = extract_cmy_channels(img);

    // Step 2: Shift each channel by offset in different directions (120° apart)
    let c_shifted = if spread_offset > 0 {
        shift_channel(&c_mask, spread_offset, spread_angle)
    } else {
        c_mask.clone()
    };

    let m_shifted = if spread_offset > 0 {
        shift_channel(&m_mask, spread_offset, spread_angle + 120.0)
    } else {
        m_mask.clone()
    };

    let y_shifted = if spread_offset > 0 {
        shift_channel(&y_mask, spread_offset, spread_angle + 240.0)
    } else {
        y_mask.clone()
    };

    // Step 3: Spread each shifted channel independently
    let c_spread = spread_channel(&c_shifted, spread_radius);
    let m_spread = spread_channel(&m_shifted, spread_radius);
    let y_spread = spread_channel(&y_shifted, spread_radius);

    // Step 4: Recombine CMY back to RGB
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let c = c_spread.get_pixel(x, y)[0] > 128;
            let m = m_spread.get_pixel(x, y)[0] > 128;
            let y_cmy = y_spread.get_pixel(x, y)[0] > 128;

            // Subtractive color mixing: start with white, subtract each channel
            // C subtracts red, M subtracts green, Y subtracts blue
            // Overlaps naturally create darker colors (R+G=Yellow, G+B=Cyan, R+B=Magenta, R+G+B=Black)
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

/// Shift a channel mask by an offset in a given direction
fn shift_channel(mask: &ImageBuffer<Rgb<u8>, Vec<u8>>, offset: i32, angle_degrees: f32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = mask.dimensions();
    let mut output = ImageBuffer::new(width, height);

    // Convert angle to radians
    let angle_rad = angle_degrees.to_radians();
    let dx = (angle_rad.cos() * offset as f32).round() as i32;
    let dy = (angle_rad.sin() * offset as f32).round() as i32;

    for y in 0..height {
        for x in 0..width {
            // Sample from shifted position
            let src_x = (x as i32 - dx).clamp(0, width as i32 - 1) as u32;
            let src_y = (y as i32 - dy).clamp(0, height as i32 - 1) as u32;

            let pixel = mask.get_pixel(src_x, src_y);
            output.put_pixel(x, y, *pixel);
        }
    }

    output
}

/// Spread a single channel mask using circular dilation
/// Input: grayscale image where 255 = channel on, 0 = channel off
fn spread_channel(mask: &ImageBuffer<Rgb<u8>, Vec<u8>>, radius: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    if radius == 0 {
        return mask.clone();
    }

    let (width, height) = mask.dimensions();
    let mut output = ImageBuffer::new(width, height);
    let radius_f = radius as f32;
    let radius_sq = (radius_f * radius_f) as i32;

    for y in 0..height {
        for x in 0..width {
            let mut is_on = false;

            // Check all pixels within circular radius
            let y_min = (y as i32 - radius as i32).max(0) as u32;
            let y_max = (y as i32 + radius as i32).min(height as i32 - 1) as u32;
            let x_min = (x as i32 - radius as i32).max(0) as u32;
            let x_max = (x as i32 + radius as i32).min(width as i32 - 1) as u32;

            for ny in y_min..=y_max {
                for nx in x_min..=x_max {
                    // Check if within circular distance
                    let dx = nx as i32 - x as i32;
                    let dy = ny as i32 - y as i32;
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq <= radius_sq {
                        let pixel = mask.get_pixel(nx, ny)[0];
                        if pixel > 128 {
                            is_on = true;
                            break;
                        }
                    }
                }
                if is_on {
                    break;
                }
            }

            let value = if is_on { 255 } else { 0 };
            output.put_pixel(x, y, Rgb([value, value, value]));
        }
    }

    output
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

            // Screen blend: 1 - (1 - dithered) * (1 - cmy)
            // This prevents colors from getting too dark while preserving K
            let cmy_color = posterized_cmy.get_pixel(x, y);
            let cmy_r = cmy_color[0] as f64 / 255.0;
            let cmy_g = cmy_color[1] as f64 / 255.0;
            let cmy_b = cmy_color[2] as f64 / 255.0;

            let r = (1.0 - (1.0 - new_val) * (1.0 - cmy_r)) * 255.0;
            let g = (1.0 - (1.0 - new_val) * (1.0 - cmy_g)) * 255.0;
            let b = (1.0 - (1.0 - new_val) * (1.0 - cmy_b)) * 255.0;

            output.put_pixel(x, y, Rgb([r as u8, g as u8, b as u8]));

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
