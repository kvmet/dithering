mod filters;
mod kernel_optimizer;
mod kernel_expander;

use clap::Parser;
use filters::{
    apply_threshold_kernel_normalized_perceptual, apply_threshold_kernel_perceptual,
    apply_threshold_kernel_normalized_perceptual_color, apply_threshold_kernel_perceptual_color,
    apply_threshold_kernel_normalized_perceptual_color_dominant,
    apply_threshold_kernel_perceptual_color_dominant,
    apply_threshold_kernel_normalized_perceptual_color_exclusive,
    apply_threshold_kernel_perceptual_color_exclusive,
    apply_threshold_kernel_normalized_perceptual_color_cmyk,
    apply_threshold_kernel_perceptual_color_cmyk, save_as_1bit_png, save_as_color_png,
    ThresholdKernel,
};
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
        help = "Color mode: bw, color, dominant, exclusive, cmyk"
    )]
    mode: String,

    /// Dither kernel
    #[arg(
        short,
        long,
        value_name = "KERNEL",
        default_value = "optimized",
        help = "Kernel: example, optimized, bayer2, bayer4, seeded (3x3:SEED, 4x4:SEED, 3x3c:SEED, 4x4c:SEED), or annealed (anneal:3x3[:SEED], anneal:4x4[:SEED], anneal:5x5[:SEED] - omit seed for random, seeds can be alphanumeric)"
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
    let img = image::open(&cli.input).unwrap_or_else(|e| {
        eprintln!("Failed to open image '{}': {}", cli.input, e);
        std::process::exit(1);
    });

    // Parse kernel
    let kernel = if cli.kernel.contains(':') {
        let parts: Vec<&str> = cli.kernel.split(':').collect();

        // Check if it's an annealed kernel: "anneal:3x3:abc123", "anneal:4x4:xyz", or "anneal:4x4" (random)
        if (parts.len() == 3 || parts.len() == 2) && parts[0] == "anneal" {
            let size_part = parts[1];

            // If no seed provided, generate a short random alphanumeric seed
            let (seed, _seed_str) = if parts.len() == 3 {
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

            match size_part {
                "3x3" => ThresholdKernel::from_annealing(3, 3, seed),
                "4x4" => ThresholdKernel::from_annealing(4, 4, seed),
                "5x5" => ThresholdKernel::from_annealing(5, 5, seed),
                _ => {
                    eprintln!("Invalid kernel size for annealing: {}", size_part);
                    eprintln!("Available sizes: 3x3, 4x4, 5x5");
                    std::process::exit(1);
                }
            }
        } else if parts.len() == 2 {
            // Parse seeded kernel: "3x3:12345", "4x4:67890", "3x3c:42.5", "4x4c:10.7"
            let size_part = parts[0];
            let seed_part = parts[1];

            // Check if it's continuous (ends with 'c')
            if size_part.ends_with('c') {
            // Continuous seed (float)
                let seed: f32 = seed_part.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid continuous seed number: {}", seed_part);
                    std::process::exit(1);
                });

                let size_str = &size_part[..size_part.len()-1]; // Remove 'c'
                match size_str {
                    "3x3" => ThresholdKernel::from_seed_continuous(3, 3, seed),
                    "4x4" => ThresholdKernel::from_seed_continuous(4, 4, seed),
                    _ => {
                        eprintln!("Invalid kernel size: {}", size_part);
                        eprintln!("Available sizes: 3x3c, 4x4c");
                        std::process::exit(1);
                    }
                }
            } else {
                // Discrete seed (integer)
                let seed: u64 = seed_part.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid discrete seed number: {}", seed_part);
                    std::process::exit(1);
                });

                match size_part {
                    "3x3" => ThresholdKernel::from_seed(3, 3, seed),
                    "4x4" => ThresholdKernel::from_seed(4, 4, seed),
                    _ => {
                        eprintln!("Invalid kernel size: {}", size_part);
                        eprintln!("Available sizes: 3x3, 4x4");
                        std::process::exit(1);
                    }
                }
            }
        } else {
            eprintln!("Invalid kernel format: {}", cli.kernel);
            eprintln!("Use format: 3x3:SEED, 4x4:SEED, 3x3c:SEED, 4x4c:SEED, or anneal:SIZE[:SEED]");
            eprintln!("Examples: 3x3:42 (discrete), 3x3c:42.5 (continuous), anneal:4x4:abc123, anneal:5x5 (random)");
            std::process::exit(1);
        }
    } else {
        match cli.kernel.as_str() {
            "example" => ThresholdKernel::example_3x3(),
            "optimized" => ThresholdKernel::optimized_3x3(),
            "bayer2" => ThresholdKernel::bayer_2x2(),
            "bayer4" => ThresholdKernel::bayer_4x4(),
            _ => {
                eprintln!("Unknown kernel type: {}", cli.kernel);
                eprintln!("Available: example, optimized, bayer2, bayer4");
                eprintln!("Or seeded: 3x3:SEED, 4x4:SEED (discrete)");
                eprintln!("Or continuous: 3x3c:SEED, 4x4c:SEED (continuous)");
                eprintln!("Or annealed: anneal:3x3[:SEED], anneal:4x4[:SEED], anneal:5x5[:SEED] (omit seed for random, seeds can be alphanumeric)");
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
        "cmyk" => {
            if cli.normalize {
                println!(
                    "Processing {} with {} kernel (CMYK mode, normalized + perceptual, gamma={:.2})...",
                    cli.input, cli.kernel, gamma
                );
                let output =
                    apply_threshold_kernel_normalized_perceptual_color_cmyk(&img, &kernel, gamma);
                save_as_color_png(&output, &cli.output).unwrap_or_else(|e| {
                    eprintln!("Failed to save color PNG '{}': {}", cli.output, e);
                    std::process::exit(1);
                });
            } else {
                println!(
                    "Processing {} with {} kernel (CMYK mode, perceptual, gamma={:.2})...",
                    cli.input, cli.kernel, gamma
                );
                let output =
                    apply_threshold_kernel_perceptual_color_cmyk(&img, &kernel, gamma);
                save_as_color_png(&output, &cli.output).unwrap_or_else(|e| {
                    eprintln!("Failed to save color PNG '{}': {}", cli.output, e);
                    std::process::exit(1);
                });
            }
            // Measure output mean brightness
            let output_img = image::open(&cli.output).unwrap();
            let output_mean = filters::measure_mean_brightness(&output_img, true);
            println!("Input mean: {:.3}, Output mean: {:.3} (gamma: {:.3})", input_mean, output_mean, gamma);
            println!("Saved CMYK-mode PNG to {}", cli.output);
        }
        _ => {
            eprintln!("Unknown mode: {}", cli.mode);
            eprintln!("Available modes: bw, color, dominant, exclusive, cmyk");
            std::process::exit(1);
        }
    }
}
