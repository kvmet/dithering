mod filters;
mod kernel_optimizer;
mod kernel_expander;
mod posterize;
mod kernel_cache;

use clap::Parser;
use filters::{
    apply_threshold_kernel_normalized_perceptual, apply_threshold_kernel_perceptual,
    apply_threshold_kernel_normalized_perceptual_color, apply_threshold_kernel_perceptual_color,
    apply_threshold_kernel_normalized_perceptual_color_dominant,
    apply_threshold_kernel_perceptual_color_dominant,
    apply_threshold_kernel_normalized_perceptual_color_exclusive,
    apply_threshold_kernel_perceptual_color_exclusive,
    save_as_1bit_png, save_as_color_png,
    ThresholdKernel,
};
use posterize::{posterize_rgb_spread, combine_rgb_with_dithered_k};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Parser)]
#[command(name = "image_filters")]
#[command(author, version, about = "Apply dithering filters to images", long_about = None)]
struct Cli {
    /// Input image path
    #[arg(value_name = "INPUT")]
    input: String,

    /// Output image path
    #[arg(value_name = "OUTPUT")]
    output: String,

    /// Color mode
    #[arg(
        short,
        long,
        value_name = "MODE",
        default_value = "bw",
        help = "Color mode: bw, color, dominant, exclusive, posterize"
    )]
    mode: String,

    /// Dither kernel
    #[arg(
        short,
        long,
        value_name = "KERNEL",
        default_value = "anneal:4x4",
        help = "Kernel: bayer2, bayer4, or annealed (anneal:3x3[:SEED], anneal:4x4[:SEED], anneal:5x5[:SEED], anneal:6x6[:SEED], anneal:pre[:SIZE] - omit seed for random, use 'pre' to select from cache, seeds can be alphanumeric)"
    )]
    kernel: String,

    /// Tie threshold for dominant mode
    #[arg(
        short,
        long,
        value_name = "FLOAT",
        default_value = "0.3",
        help = "Tie threshold for dominant mode (0.0-1.0)"
    )]
    threshold: f32,

    /// Enable percentile normalization
    #[arg(short, long, help = "Enable percentile normalization")]
    normalize: bool,

    /// Gamma correction (auto by default)
    #[arg(
        short,
        long,
        value_name = "FLOAT",
        help = "Gamma correction (< 1.0 = darker, > 1.0 = brighter). Auto-computed if not specified."
    )]
    gamma: Option<f32>,

    /// Auto-gamma offset multiplier
    #[arg(
        long,
        value_name = "FLOAT",
        default_value = "0.8",
        help = "Multiplier for auto-computed gamma (compensates for sRGB→linear darkening)"
    )]
    auto_gamma_offset: f32,

    /// Print kernel values and exit (use with -k to specify kernel)
    #[arg(long, help = "Print kernel values and exit")]
    print_kernel: bool,

    /// Print score table and exit (use with -k to specify kernel)
    #[arg(long, help = "Print score table and exit")]
    print_scores: bool,

    /// Target width (scales height proportionally if height not specified)
    #[arg(short = 'W', long, value_name = "PIXELS", help = "Target width in pixels")]
    width: Option<u32>,

    /// Target height (scales width proportionally if width not specified)
    #[arg(short = 'H', long, value_name = "PIXELS", help = "Target height in pixels")]
    height: Option<u32>,

    /// Spread radius for posterize mode (grows CMY color areas)
    #[arg(long, value_name = "PIXELS", default_value = "3", help = "Spread radius for posterize mode (pixels)")]
    spread_radius: u32,

    /// Spread offset for posterize mode (shifts CMY channels before spreading)
    #[arg(long, value_name = "PIXELS", default_value = "0", help = "Offset distance for CMY channel shifting")]
    spread_offset: i32,

    /// Spread angle for posterize mode (direction to shift CMY channels, in degrees)
    #[arg(long, value_name = "DEGREES", default_value = "0.0", help = "Angle for CMY channel shifting (0=right, 90=down, etc.)")]
    spread_angle: f32,

    /// Erode radius for posterize mode (shrinks colors after spreading to round corners)
    #[arg(long, value_name = "PIXELS", default_value = "0", help = "Erode radius for posterize mode (rounds corners)")]
    erode_radius: u32,

    /// Color threshold for posterize mode (minimum brightness to be considered "on", 0.0-1.0)
    #[arg(long, value_name = "COLOR_THRESHOLD", default_value = "0.2", help = "Color threshold for posterize mode (0.0-1.0)")]
    color_threshold: f32,

    /// White threshold for posterize mode (brightness above which all channels = white/no color, 0.0-1.0)
    #[arg(long, value_name = "WHITE_THRESHOLD", default_value = "0.9", help = "White threshold for posterize mode (0.0-1.0)")]
    white_threshold: f32,

    /// Color intensity for posterize mode (screen blend intensity, 0.0=white, 1.0=full color)
    #[arg(long, value_name = "INTENSITY", default_value = "1.0", help = "Color intensity for posterize mode (0.0-1.0)")]
    color_intensity: f32,

    /// Vertical adjacency weight for kernel optimization
    #[arg(long, value_name = "FLOAT", help = "Vertical adjacency weight for kernel optimization [default: 0.3]")]
    weight_vertical: Option<f64>,

    /// Horizontal adjacency weight for kernel optimization
    #[arg(long, value_name = "FLOAT", help = "Horizontal adjacency weight for kernel optimization [default: 0.3]")]
    weight_horizontal: Option<f64>,

    /// Positive diagonal weight for kernel optimization
    #[arg(long, value_name = "FLOAT", help = "Positive diagonal weight for kernel optimization [default: 0.5]")]
    weight_diagonal_pos: Option<f64>,

    /// Negative diagonal weight for kernel optimization
    #[arg(long, value_name = "FLOAT", help = "Negative diagonal weight for kernel optimization [default: 0.5]")]
    weight_diagonal_neg: Option<f64>,

    /// Knight move weight for kernel optimization
    #[arg(long, value_name = "FLOAT", help = "Knight move weight for kernel optimization [default: 0.7]")]
    weight_knight: Option<f64>,

    /// Other relationships weight for kernel optimization
    #[arg(long, value_name = "FLOAT", help = "Other relationships weight for kernel optimization [default: 1.0]")]
    weight_other: Option<f64>,
}

fn main() {
    let cli = Cli::parse();

    // Validate threshold range
    if cli.threshold < 0.0 || cli.threshold > 1.0 {
        eprintln!("Error: Tie threshold must be between 0.0 and 1.0");
        std::process::exit(1);
    }

    // Validate gamma if provided
    if let Some(gamma) = cli.gamma {
        if gamma <= 0.0 {
            eprintln!("Error: Gamma must be greater than 0.0");
            std::process::exit(1);
        }
    }

    // Validate auto-gamma offset
    if cli.auto_gamma_offset <= 0.0 {
        eprintln!("Error: Auto-gamma offset must be greater than 0.0");
        std::process::exit(1);
    }

    // Load the image
    let mut img = image::open(&cli.input).unwrap_or_else(|e| {
        eprintln!("Failed to open image '{}': {}", cli.input, e);
        std::process::exit(1);
    });

    // Resize image if width/height specified
    if cli.width.is_some() || cli.height.is_some() {
        let (orig_w, orig_h) = (img.width(), img.height());
        let (target_w, target_h) = match (cli.width, cli.height) {
            (Some(w), Some(h)) => {
                // Both specified: stretch to exact dimensions
                (w, h)
            }
            (Some(w), None) => {
                // Only width: maintain aspect ratio
                let h = ((w as f64 / orig_w as f64) * orig_h as f64).round() as u32;
                (w, h)
            }
            (None, Some(h)) => {
                // Only height: maintain aspect ratio
                let w = ((h as f64 / orig_h as f64) * orig_w as f64).round() as u32;
                (w, h)
            }
            (None, None) => unreachable!(),
        };

        if target_w != orig_w || target_h != orig_h {
            println!("Resizing from {}x{} to {}x{}", orig_w, orig_h, target_w, target_h);
            img = img.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);
        }
    }

    // Parse kernel
    let kernel = if cli.kernel.contains(':') {
        let parts: Vec<&str> = cli.kernel.split(':').collect();

        // Check if it's an annealed kernel: "anneal:4x4:abc123", "anneal:4x4:xyz", or "anneal:4x4" (random)
        if (parts.len() == 3 || parts.len() == 2) && parts[0] == "anneal" {
            let size_part = parts[1];

            // Check for "pre" to select from cached kernels
            if size_part == "pre" {
                // Determine size from parts[2] if provided, otherwise error
                let size = if parts.len() == 3 {
                    match parts[2] {
                        "3x3" => 3,
                        "4x4" => 4,
                        "5x5" => 5,
                        "6x6" => 6,
                        _ => {
                            eprintln!("Invalid kernel size: {}", parts[2]);
                            eprintln!("Available sizes: 3x3, 4x4, 5x5, 6x6");
                            std::process::exit(1);
                        }
                    }
                } else {
                    // Default to 4x4 if no size specified
                    4
                };

                let cached = kernel_cache::list_cached_kernels(size);
                if cached.is_empty() {
                    eprintln!("No cached {}x{} kernels found in kernels/ directory", size, size);
                    eprintln!("Generate some first with: anneal:{}x{}", size, size);
                    std::process::exit(1);
                }

                // Pick a random one
                use std::time::{SystemTime, UNIX_EPOCH};
                let rand_idx = (SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() % cached.len() as u128) as usize;

                let (seed, seed_string) = &cached[rand_idx];
                let seed_owned = seed.to_string();
                let display_seed = seed_string.as_deref().unwrap_or(&seed_owned);
                println!("Selected cached kernel: {}x{} seed {}", size, size, display_seed);

                ThresholdKernel::from_annealing_with_string(size, size, *seed, seed_string.clone())
            } else {
                // Original behavior: size is specified like "4x4"
                // Build geometry weights if any are specified
                let custom_weights = if cli.weight_vertical.is_some()
                    || cli.weight_horizontal.is_some()
                    || cli.weight_diagonal_pos.is_some()
                    || cli.weight_diagonal_neg.is_some()
                    || cli.weight_knight.is_some()
                    || cli.weight_other.is_some() {

                    let defaults = kernel_optimizer::GeometryWeights::default();
                    Some(kernel_optimizer::GeometryWeights {
                        vertical: cli.weight_vertical.unwrap_or(defaults.vertical),
                        horizontal: cli.weight_horizontal.unwrap_or(defaults.horizontal),
                        positive_diagonal: cli.weight_diagonal_pos.unwrap_or(defaults.positive_diagonal),
                        negative_diagonal: cli.weight_diagonal_neg.unwrap_or(defaults.negative_diagonal),
                        knight: cli.weight_knight.unwrap_or(defaults.knight),
                        other: cli.weight_other.unwrap_or(defaults.other),
                    })
                } else {
                    None
                };

                // If no seed provided, generate a short random alphanumeric seed
            let (seed, seed_str) = if parts.len() == 3 {
                let seed_str = parts[2].to_string();
                // Try to parse as number first, otherwise hash the string
                let seed = seed_str.parse::<u64>().unwrap_or_else(|_| {
                    ThresholdKernel::seed_from_string(&seed_str)
                });
                (seed, seed_str)
            } else {
                // Generate short alphanumeric seed (6 characters)

                let time_seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;

                // Convert to base-36 (0-9, a-z) for shorter representation
                let mut seed_str = String::new();
                let mut n = time_seed;
                let chars = "0123456789abcdefghijklmnopqrstuvwxyz";
                for _ in 0..6 {
                    let idx = (n % 36) as usize;
                    seed_str.push(chars.chars().nth(idx).unwrap());
                    n /= 36;
                }

                println!("Using random seed: {}", seed_str);
                (time_seed, seed_str)
            };

                let size = match size_part {
                    "3x3" => 3,
                    "4x4" => 4,
                    "5x5" => 5,
                    "6x6" => 6,
                    _ => {
                        eprintln!("Invalid kernel size for annealing: {}", size_part);
                        eprintln!("Available sizes: 3x3, 4x4, 5x5, 6x6, or 'pre' for cached");
                        std::process::exit(1);
                    }
                };

                // Generate kernel with custom weights if provided
                if let Some(weights) = custom_weights {
                    ThresholdKernel::from_annealing_with_weights(size, size, seed, Some(seed_str), weights)
                } else {
                    ThresholdKernel::from_annealing_with_string(size, size, seed, Some(seed_str))
                }
            }
        } else {
            eprintln!("Invalid kernel format: {}", cli.kernel);
            eprintln!("Use format: anneal:SIZE[:SEED] or anneal:pre[:SIZE]");
            eprintln!("Examples: anneal:4x4:abc123, anneal:5x5 (random), anneal:pre:4x4, anneal:pre");
            std::process::exit(1);
        }
    } else {
        match cli.kernel.as_str() {
            "bayer2" => ThresholdKernel::bayer_2x2(),
            "bayer4" => ThresholdKernel::bayer_4x4(),
            _ => {
                eprintln!("Unknown kernel type: {}", cli.kernel);
                eprintln!("Available: bayer2, bayer4");
                eprintln!("Or annealed: anneal:3x3[:SEED], anneal:4x4[:SEED], anneal:5x5[:SEED], anneal:6x6[:SEED] (omit seed for random, seeds can be alphanumeric)");
                std::process::exit(1);
            }
        }
    };

    // If --print-kernel flag is set, print kernel and exit
    if cli.print_kernel {
        println!("Kernel: {} ({}x{})", cli.kernel, kernel.width, kernel.height);
        println!();
        for y in 0..kernel.height {
            for x in 0..kernel.width {
                print!("{:6.4}  ", kernel.get(x, y));
            }
            println!();
        }
        println!();
        println!("Values (flat): {:?}", kernel.values);

        // Compute and display score
        let kernel_obj = kernel_optimizer::Kernel::new(kernel.width, kernel.values.clone());
        let positions = kernel_obj.build_positions();
        let scorer = kernel_optimizer::IncrementalScorer::new(kernel.width, positions);
        let score = scorer.total_score();
        println!("Score: {:.2}", score);

        return;
    }

    // If --print-scores flag is set, print score table and exit
    if cli.print_scores {
        println!("Kernel: {} ({}x{})", cli.kernel, kernel.width, kernel.height);
        println!();

        // Build the kernel and compute scoring table
        use kernel_optimizer::Kernel;
        let kernel_obj = Kernel::new(kernel.width, kernel.values.clone());
        let positions = kernel_obj.build_positions();
        let scorer = kernel_optimizer::IncrementalScorer::new(kernel.width, positions);
        let score = scorer.total_score();

        println!("Overall Score: {:.2}", score);
        println!();

        // Create the score lookup table
        use kernel_optimizer::ScoreLookup;
        let lookup = ScoreLookup::new(kernel.width);

        // Print score matrices for various value distances
        println!("Position-to-Position Score Matrices:");
        println!("(Shows how much each position pair contributes to the score for different value distances)");
        println!();

        // Print matrices for a few representative value distances
        let n = kernel.values.len();
        let distances_to_show = vec![1, n / 4, n / 2, n - 1];

        for &dist in &distances_to_show {
            if dist > 0 && dist < n {
                // Use value indices 0 and dist to demonstrate the score matrix for this distance
                lookup.print_score_matrix(dist, 0, dist);
            }
        }

        return;
    }

    // Measure input mean brightness
    let input_mean = filters::measure_mean_brightness(&img, true);

    // Determine gamma to use
    let gamma = cli.gamma.unwrap_or_else(|| {
        if cli.normalize {
            // Normalization already maximizes dynamic range, use neutral gamma
            1.0
        } else {
            // Auto-compute gamma to match input/output brightness
            // Apply offset to compensate for sRGB→linear darkening
            let auto_gamma = filters::compute_auto_gamma(&img, &kernel, true, false);
            let adjusted_gamma = auto_gamma * cli.auto_gamma_offset;
            println!("Auto-computed gamma: {:.3} (base: {:.3}, offset: {:.2})", adjusted_gamma, auto_gamma, cli.auto_gamma_offset);
            adjusted_gamma
        }
    });

    // Process based on user-selected mode
    match cli.mode.as_str() {
        "color" => {
            if cli.normalize {
                println!(
                    "Processing {} with {} kernel (color, normalized + perceptual, gamma={:.2})...",
                    cli.input, cli.kernel, gamma
                );
                let output =
                    apply_threshold_kernel_normalized_perceptual_color(&img, &kernel, gamma);
                save_as_color_png(&output, &cli.output).unwrap_or_else(|e| {
                    eprintln!("Failed to save color PNG '{}': {}", cli.output, e);
                    std::process::exit(1);
                });
            } else {
                println!(
                    "Processing {} with {} kernel (color, perceptual, gamma={:.2})...",
                    cli.input, cli.kernel, gamma
                );
                let output = apply_threshold_kernel_perceptual_color(&img, &kernel, gamma);
                save_as_color_png(&output, &cli.output).unwrap_or_else(|e| {
                    eprintln!("Failed to save color PNG '{}': {}", cli.output, e);
                    std::process::exit(1);
                });
            }
            // Measure output mean brightness
            let output_img = image::open(&cli.output).unwrap();
            let output_mean = filters::measure_mean_brightness(&output_img, true);
            println!("Input mean: {:.3}, Output mean: {:.3} (gamma: {:.3})", input_mean, output_mean, gamma);
            println!("Saved 8-color PNG to {}", cli.output);
        }
        "dominant" => {
            if cli.normalize {
                println!(
                    "Processing {} with {} kernel (dominant channel, tie_threshold={}, normalized + perceptual, gamma={:.2})...",
                    cli.input, cli.kernel, cli.threshold, gamma
                );
                let output = apply_threshold_kernel_normalized_perceptual_color_dominant(
                    &img,
                    &kernel,
                    cli.threshold,
                    gamma,
                );
                save_as_color_png(&output, &cli.output).unwrap_or_else(|e| {
                    eprintln!("Failed to save color PNG '{}': {}", cli.output, e);
                    std::process::exit(1);
                });
            } else {
                println!(
                    "Processing {} with {} kernel (dominant channel, tie_threshold={}, perceptual, gamma={:.2})...",
                    cli.input, cli.kernel, cli.threshold, gamma
                );
                let output = apply_threshold_kernel_perceptual_color_dominant(
                    &img,
                    &kernel,
                    cli.threshold,
                    gamma,
                );
                save_as_color_png(&output, &cli.output).unwrap_or_else(|e| {
                    eprintln!("Failed to save color PNG '{}': {}", cli.output, e);
                    std::process::exit(1);
                });
            }
            // Measure output mean brightness
            let output_img = image::open(&cli.output).unwrap();
            let output_mean = filters::measure_mean_brightness(&output_img, true);
            println!("Input mean: {:.3}, Output mean: {:.3} (gamma: {:.3})", input_mean, output_mean, gamma);
            println!("Saved dominant-channel PNG to {}", cli.output);
        }
        "exclusive" => {
            if cli.normalize {
                println!(
                    "Processing {} with {} kernel (exclusive mode, normalized + perceptual, gamma={:.2})...",
                    cli.input, cli.kernel, gamma
                );
                let output =
                    apply_threshold_kernel_normalized_perceptual_color_exclusive(&img, &kernel, gamma);
                save_as_color_png(&output, &cli.output).unwrap_or_else(|e| {
                    eprintln!("Failed to save color PNG '{}': {}", cli.output, e);
                    std::process::exit(1);
                });
            } else {
                println!(
                    "Processing {} with {} kernel (exclusive mode, perceptual, gamma={:.2})...",
                    cli.input, cli.kernel, gamma
                );
                let output =
                    apply_threshold_kernel_perceptual_color_exclusive(&img, &kernel, gamma);
                save_as_color_png(&output, &cli.output).unwrap_or_else(|e| {
                    eprintln!("Failed to save color PNG '{}': {}", cli.output, e);
                    std::process::exit(1);
                });
            }
            // Measure output mean brightness
            let output_img = image::open(&cli.output).unwrap();
            let output_mean = filters::measure_mean_brightness(&output_img, true);
            println!("Input mean: {:.3}, Output mean: {:.3} (gamma: {:.3})", input_mean, output_mean, gamma);
            println!("Saved exclusive-mode PNG to {}", cli.output);
        }
        "bw" => {
            if cli.normalize {
                println!(
                    "Processing {} with {} kernel (monochrome, normalized + perceptual, gamma={:.2})...",
                    cli.input, cli.kernel, gamma
                );
                let output =
                    apply_threshold_kernel_normalized_perceptual(&img, &kernel, gamma);
                save_as_1bit_png(&output, &cli.output).unwrap_or_else(|e| {
                    eprintln!("Failed to save 1-bit PNG '{}': {}", cli.output, e);
                    std::process::exit(1);
                });
            } else {
                println!(
                    "Processing {} with {} kernel (monochrome, perceptual, gamma={:.2})...",
                    cli.input, cli.kernel, gamma
                );
                let output = apply_threshold_kernel_perceptual(&img, &kernel, gamma);
                save_as_1bit_png(&output, &cli.output).unwrap_or_else(|e| {
                    eprintln!("Failed to save 1-bit PNG '{}': {}", cli.output, e);
                    std::process::exit(1);
                });
            }
            // Measure output mean brightness
            let output_img = image::open(&cli.output).unwrap();
            let output_mean = filters::measure_mean_brightness(&output_img, true);
            println!("Input mean: {:.3}, Output mean: {:.3} (gamma: {:.3})", input_mean, output_mean, gamma);
            println!("Saved 1-bit PNG to {}", cli.output);
        }
        "posterize" => {
            println!(
                "Processing {} with {} kernel (posterize mode, spread={}, offset={}, angle={:.1}°, erode={}, color_thresh={:.2}, white_thresh={:.2}, intensity={:.2}, gamma={:.2})...",
                cli.input, cli.kernel, cli.spread_radius, cli.spread_offset, cli.spread_angle, cli.erode_radius, cli.color_threshold, cli.white_threshold, cli.color_intensity, gamma
            );
            let posterized_rgb = posterize_rgb_spread(&img, cli.spread_radius, cli.spread_offset, cli.spread_angle, cli.erode_radius, cli.color_threshold, cli.white_threshold, cli.color_intensity);
            let output = combine_rgb_with_dithered_k(&posterized_rgb, &img, &kernel, gamma);
            save_as_color_png(&output, &cli.output).unwrap_or_else(|e| {
                eprintln!("Failed to save color PNG '{}': {}", cli.output, e);
                std::process::exit(1);
            });
            // Measure output mean brightness
            let output_img = image::open(&cli.output).unwrap();
            let output_mean = filters::measure_mean_brightness(&output_img, true);
            println!("Input mean: {:.3}, Output mean: {:.3} (gamma: {:.3})", input_mean, output_mean, gamma);
            println!("Saved posterized PNG to {}", cli.output);
        }
        _ => {
            eprintln!("Unknown mode: {}", cli.mode);
            eprintln!("Available modes: bw, color, dominant, exclusive, posterize");
            std::process::exit(1);
        }
    }
}
