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
pub fn posterize_rgb_spread(
    img: &DynamicImage,
    spread_radius: u32,
    spread_offset: i32,
    spread_angle: f32,
    erode_radius: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    // Step 1: Extract R, G, B channels as separate boolean masks
    let (r_mask, g_mask, b_mask) = extract_rgb_channels(img);

    // Step 2: Shift each channel by offset in different directions (120° apart)
    let r_shifted = if spread_offset > 0 {
        shift_channel(&r_mask, spread_offset, spread_angle)
    } else {
        r_mask.clone()
    };

    let g_shifted = if spread_offset > 0 {
        shift_channel(&g_mask, spread_offset, spread_angle + 120.0)
    } else {
        g_mask.clone()
    };

    let b_shifted = if spread_offset > 0 {
        shift_channel(&b_mask, spread_offset, spread_angle + 240.0)
    } else {
        b_mask.clone()
    };

    // Step 3: Spread each shifted channel independently
    let r_spread = spread_channel(&r_shifted, spread_radius);
    let g_spread = spread_channel(&g_shifted, spread_radius);
    let b_spread = spread_channel(&b_shifted, spread_radius);

    // Step 3.5: Optionally erode to round corners (morphological closing)
    let r_final = if erode_radius > 0 {
        erode_channel(&r_spread, erode_radius)
    } else {
        r_spread
    };
    let g_final = if erode_radius > 0 {
        erode_channel(&g_spread, erode_radius)
    } else {
        g_spread
    };
    let b_final = if erode_radius > 0 {
        erode_channel(&b_spread, erode_radius)
    } else {
        b_spread
    };

    // Step 4: Recombine RGB with reflect blending
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let r_on = r_final.get_pixel(x, y)[0] > 128;
            let g_on = g_final.get_pixel(x, y)[0] > 128;
            let b_on = b_final.get_pixel(x, y)[0] > 128;

            // Screen blend: 1 - (1 - a) * (1 - b)
            // Start with black (0, 0, 0) and screen blend each active channel
            let mut result_r = 0.0f32;
            let mut result_g = 0.0f32;
            let mut result_b = 0.0f32;

            if r_on {
                // Blend in pure red (1, 0, 0)
                result_r = 1.0 - (1.0 - result_r) * (1.0 - 1.0);
                result_g = 1.0 - (1.0 - result_g) * (1.0 - 0.0);
                result_b = 1.0 - (1.0 - result_b) * (1.0 - 0.0);
            }
            if g_on {
                // Blend in pure green (0, 1, 0)
                result_r = 1.0 - (1.0 - result_r) * (1.0 - 0.0);
                result_g = 1.0 - (1.0 - result_g) * (1.0 - 1.0);
                result_b = 1.0 - (1.0 - result_b) * (1.0 - 0.0);
            }
            if b_on {
                // Blend in pure blue (0, 0, 1)
                result_r = 1.0 - (1.0 - result_r) * (1.0 - 0.0);
                result_g = 1.0 - (1.0 - result_g) * (1.0 - 0.0);
                result_b = 1.0 - (1.0 - result_b) * (1.0 - 1.0);
            }

            // If no channels are active, make it white instead of black
            if !r_on && !g_on && !b_on {
                result_r = 1.0;
                result_g = 1.0;
                result_b = 1.0;
            }

            let r = (result_r * 255.0) as u8;
            let g = (result_g * 255.0) as u8;
            let b = (result_b * 255.0) as u8;

            output.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    output
}

/// Extract R, G, B channels as separate boolean masks
/// Returns (R_mask, G_mask, B_mask) where 255 = channel is on, 0 = channel is off
fn extract_rgb_channels(img: &DynamicImage) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, ImageBuffer<Rgb<u8>, Vec<u8>>, ImageBuffer<Rgb<u8>, Vec<u8>>) {
    let (width, height) = img.dimensions();
    let mut r_mask = ImageBuffer::new(width, height);
    let mut g_mask = ImageBuffer::new(width, height);
    let mut b_mask = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            // Threshold each channel independently (>0.5 = on)
            let r_on = if r > 0.5 { 255 } else { 0 };
            let g_on = if g > 0.5 { 255 } else { 0 };
            let b_on = if b > 0.5 { 255 } else { 0 };

            r_mask.put_pixel(x, y, Rgb([r_on, r_on, r_on]));
            g_mask.put_pixel(x, y, Rgb([g_on, g_on, g_on]));
            b_mask.put_pixel(x, y, Rgb([b_on, b_on, b_on]));
        }
    }

    (r_mask, g_mask, b_mask)
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

/// Erode a single channel mask using circular erosion
/// Input: grayscale image where 255 = channel on, 0 = channel off
/// This shrinks white regions, rounding out corners
fn erode_channel(mask: &ImageBuffer<Rgb<u8>, Vec<u8>>, radius: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    if radius == 0 {
        return mask.clone();
    }

    let (width, height) = mask.dimensions();
    let mut output = ImageBuffer::new(width, height);
    let radius_f = radius as f32;
    let radius_sq = (radius_f * radius_f) as i32;

    for y in 0..height {
        for x in 0..width {
            let mut is_on = true;

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
                        if pixel <= 128 {
                            is_on = false;
                            break;
                        }
                    }
                }
                if !is_on {
                    break;
                }
            }

            let value = if is_on { 255 } else { 0 };
            output.put_pixel(x, y, Rgb([value, value, value]));
        }
    }

    output
}

/// Combine posterized RGB with K channel using ordered dithering
///
/// Algorithm:
/// 1. Compute luminance from original image
/// 2. Use ordered dither threshold to decide: output posterized RGB color or black
pub fn combine_rgb_with_dithered_k(
    posterized_rgb: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    img: &DynamicImage,
    kernel: &ThresholdKernel,
    gamma: f32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    let rgb_img = img.to_rgb8();
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            // Get original pixel
            let pixel = rgb_img.get_pixel(x, y);
            let r = (pixel[0] as f64 / 255.0).powf(2.2);
            let g = (pixel[1] as f64 / 255.0).powf(2.2);
            let b = (pixel[2] as f64 / 255.0).powf(2.2);

            // Compute original perceptual luminance
            let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;

            // Apply gamma correction
            let adjusted_luminance = luminance.powf(gamma as f64);

            // Get posterized RGB color for this pixel
            let rgb_color = posterized_rgb.get_pixel(x, y);

            // Get threshold from ordered dither kernel
            let threshold = kernel.get(
                (x as usize) % kernel.width,
                (y as usize) % kernel.height,
            );

            // Binary decision: if luminance > threshold, show posterized RGB color (or white), else black
            if adjusted_luminance > threshold {
                // Pixel is "on" - output the posterized RGB color
                output.put_pixel(x, y, *rgb_color);
            } else {
                // Pixel is "off" - output black
                output.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
    }

    output
}
