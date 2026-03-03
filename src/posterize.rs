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
    color_threshold: f32,
    white_threshold: f32,
    color_intensity: f32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    // Step 1: Extract R, G, B channels as separate boolean masks
    // Enforce max 2 colors per pixel (keep strongest 1-2 channels)
    let (r_mask, g_mask, b_mask) = extract_rgb_channels_max2(img, color_threshold, white_threshold);

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

    // Step 3: Spread all channels together with 2-color-max enforcement
    let (r_spread, g_spread, b_spread) = spread_channels_max2(&r_shifted, &g_shifted, &b_shifted, spread_radius);

    // Step 3.5: Optionally erode to round corners (morphological closing)
    // Erode across all channels - any color protects neighboring colors
    let (r_final, g_final, b_final) = if erode_radius > 0 {
        erode_channels_combined(&r_spread, &g_spread, &b_spread, erode_radius)
    } else {
        (r_spread, g_spread, b_spread)
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

            // Screen blend semi-transparent white overlay on top
            // intensity=1.0 -> no white overlay (full color)
            // intensity=0.0 -> full white overlay (everything becomes white)
            let white_opacity = 1.0 - color_intensity;
            result_r = 1.0 - (1.0 - result_r) * (1.0 - white_opacity);
            result_g = 1.0 - (1.0 - result_g) * (1.0 - white_opacity);
            result_b = 1.0 - (1.0 - result_b) * (1.0 - white_opacity);

            let r = (result_r * 255.0) as u8;
            let g = (result_g * 255.0) as u8;
            let b = (result_b * 255.0) as u8;

            output.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    output
}

/// Extract R, G, B channels with max 2 colors per pixel
/// Returns (R_mask, G_mask, B_mask) where 255 = channel is on, 0 = channel is off
/// Only keeps the strongest 1 or 2 channels per pixel
/// Pixels above white_threshold on all channels are treated as white (no colors)
fn extract_rgb_channels_max2(img: &DynamicImage, color_threshold: f32, white_threshold: f32) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, ImageBuffer<Rgb<u8>, Vec<u8>>, ImageBuffer<Rgb<u8>, Vec<u8>>) {
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

            // Check if this is a white pixel (all channels above white threshold)
            let is_white = r >= white_threshold && g >= white_threshold && b >= white_threshold;

            // Determine which channels are above color threshold
            let r_above = r > color_threshold && !is_white;
            let g_above = g > color_threshold && !is_white;
            let b_above = b > color_threshold && !is_white;

            let count = r_above as u8 + g_above as u8 + b_above as u8;

            let (r_on, g_on, b_on) = if count <= 2 {
                // 0, 1, or 2 channels - keep all above threshold
                (r_above, g_above, b_above)
            } else {
                // All 3 channels above threshold - keep strongest 2
                // Sort by value, keep top 2
                let mut channels = vec![
                    (r, 0), // 0 = R
                    (g, 1), // 1 = G
                    (b, 2), // 2 = B
                ];
                channels.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                let keep_r = channels[0].1 == 0 || channels[1].1 == 0;
                let keep_g = channels[0].1 == 1 || channels[1].1 == 1;
                let keep_b = channels[0].1 == 2 || channels[1].1 == 2;

                (keep_r, keep_g, keep_b)
            };

            r_mask.put_pixel(x, y, Rgb([if r_on { 255 } else { 0 }; 3]));
            g_mask.put_pixel(x, y, Rgb([if g_on { 255 } else { 0 }; 3]));
            b_mask.put_pixel(x, y, Rgb([if b_on { 255 } else { 0 }; 3]));
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

/// Spread all channels together with max 2 colors enforcement
/// Rules:
/// - Colors can only spread into pixels that share at least one color
/// - This prevents 3-color collisions naturally
/// - Max 2 colors per pixel enforced
fn spread_channels_max2(
    r_mask: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    g_mask: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    b_mask: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    radius: u32,
) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, ImageBuffer<Rgb<u8>, Vec<u8>>, ImageBuffer<Rgb<u8>, Vec<u8>>) {
    if radius == 0 {
        return (r_mask.clone(), g_mask.clone(), b_mask.clone());
    }

    let (width, height) = r_mask.dimensions();

    // Start with copies of input masks
    let mut r_current = r_mask.clone();
    let mut g_current = g_mask.clone();
    let mut b_current = b_mask.clone();

    // Iterate radius times, spreading by 1 pixel each iteration
    // This gives O(n × radius) instead of O(n × radius²)
    for _ in 0..radius {
        let mut r_next = r_current.clone();
        let mut g_next = g_current.clone();
        let mut b_next = b_current.clone();

        for y in 0..height {
            for x in 0..width {
                let current_r = r_current.get_pixel(x, y)[0] > 128;
                let current_g = g_current.get_pixel(x, y)[0] > 128;
                let current_b = b_current.get_pixel(x, y)[0] > 128;

                let mut wants_r = current_r;
                let mut wants_g = current_g;
                let mut wants_b = current_b;

                // Check 4-connected neighbors (faster than 8-connected for circular spread)
                let neighbors = [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ];

                for (nx, ny) in neighbors {
                    if nx < width && ny < height {
                        let neighbor_r = r_current.get_pixel(nx, ny)[0] > 128;
                        let neighbor_g = g_current.get_pixel(nx, ny)[0] > 128;
                        let neighbor_b = b_current.get_pixel(nx, ny)[0] > 128;

                        // Check if neighbor shares at least one color with current pixel
                        let shares_color = (neighbor_r && current_r) ||
                                          (neighbor_g && current_g) ||
                                          (neighbor_b && current_b);

                        // If current pixel has no colors, any neighbor can spread in
                        let can_spread = shares_color || (!current_r && !current_g && !current_b);

                        if can_spread {
                            if neighbor_r { wants_r = true; }
                            if neighbor_g { wants_g = true; }
                            if neighbor_b { wants_b = true; }
                        }
                    }
                }

                // Count how many colors want this pixel
                let count = wants_r as u8 + wants_g as u8 + wants_b as u8;

                // Apply max-2-color rule
                let (final_r, final_g, final_b) = if count <= 2 {
                    // 0, 1, or 2 colors - allow all
                    (wants_r, wants_g, wants_b)
                } else {
                    // All 3 colors want this pixel - keep current state
                    (current_r, current_g, current_b)
                };

                r_next.put_pixel(x, y, Rgb([if final_r { 255 } else { 0 }; 3]));
                g_next.put_pixel(x, y, Rgb([if final_g { 255 } else { 0 }; 3]));
                b_next.put_pixel(x, y, Rgb([if final_b { 255 } else { 0 }; 3]));
            }
        }

        r_current = r_next;
        g_current = g_next;
        b_current = b_next;
    }

    (r_current, g_current, b_current)
}

/// Erode all RGB channels together using combined mask
/// A pixel stays "on" in a channel only if ANY color is present nearby
/// This allows different color channels to protect each other from erosion
fn erode_channels_combined(
    r_mask: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    g_mask: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    b_mask: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    radius: u32,
) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, ImageBuffer<Rgb<u8>, Vec<u8>>, ImageBuffer<Rgb<u8>, Vec<u8>>) {
    if radius == 0 {
        return (r_mask.clone(), g_mask.clone(), b_mask.clone());
    }

    let (width, height) = r_mask.dimensions();
    let mut r_output = ImageBuffer::new(width, height);
    let mut g_output = ImageBuffer::new(width, height);
    let mut b_output = ImageBuffer::new(width, height);
    let radius_f = radius as f32;
    let radius_sq = (radius_f * radius_f) as i32;

    for y in 0..height {
        for x in 0..width {
            // Check if this pixel should stay on based on combined color presence nearby
            let mut any_color_nearby = false;

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
                        // Check if ANY channel is on at this location
                        let r_on = r_mask.get_pixel(nx, ny)[0] > 128;
                        let g_on = g_mask.get_pixel(nx, ny)[0] > 128;
                        let b_on = b_mask.get_pixel(nx, ny)[0] > 128;

                        if r_on || g_on || b_on {
                            any_color_nearby = true;
                            break;
                        }
                    }
                }
                if any_color_nearby {
                    break;
                }
            }

            // Each channel stays on only if:
            // 1. It was originally on at this pixel, AND
            // 2. There's any color nearby (protects from erosion)
            let r_was_on = r_mask.get_pixel(x, y)[0] > 128;
            let g_was_on = g_mask.get_pixel(x, y)[0] > 128;
            let b_was_on = b_mask.get_pixel(x, y)[0] > 128;

            let r_value = if r_was_on && any_color_nearby { 255 } else { 0 };
            let g_value = if g_was_on && any_color_nearby { 255 } else { 0 };
            let b_value = if b_was_on && any_color_nearby { 255 } else { 0 };

            r_output.put_pixel(x, y, Rgb([r_value, r_value, r_value]));
            g_output.put_pixel(x, y, Rgb([g_value, g_value, g_value]));
            b_output.put_pixel(x, y, Rgb([b_value, b_value, b_value]));
        }
    }

    (r_output, g_output, b_output)
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
