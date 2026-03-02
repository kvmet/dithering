# Image Filters

A CLI tool for applying dithering filters to images with various color modes and kernel options.

## Installation

```bash
cargo build --release
```

The binary will be at `target/release/image_filters`.

## Usage

```bash
image_filters [OPTIONS] <INPUT> <OUTPUT>
```

### Arguments

- `<INPUT>` - Input image path
- `<OUTPUT>` - Output image path

### Options

- `-m, --mode <MODE>` - Color mode [default: bw]
  - `bw` - Black & white (1-bit PNG)
  - `color` - 8 colors (per-channel dithering)
  - `dominant` - Dominant channel only (more saturated)
  - `exclusive` - Only R, G, B, or W (5 colors)
  - `cmyk` - Only C, M, Y, K, or W (5 colors)

- `-k, --kernel <KERNEL>` - Dither kernel [default: optimized]
  - `example` - Example 3x3 kernel
  - `optimized` - Optimized 3x3 kernel
  - `bayer2` - Bayer 2x2 pattern
  - `bayer4` - Bayer 4x4 pattern
  - `3x3:SEED` or `4x4:SEED` - Discrete random seed (e.g., `3x3:42`)
  - `3x3c:SEED` or `4x4c:SEED` - Continuous seed (e.g., `3x3c:42.5`)
  - `anneal:3x3[:SEED]` or `anneal:4x4[:SEED]` - Optimized via simulated annealing (e.g., `anneal:4x4:12345` or `anneal:4x4` for random)

- `-t, --threshold <FLOAT>` - Tie threshold for dominant mode [default: 0.3]
  - Range: 0.0-1.0
  - Only affects `dominant` mode

- `-n, --normalize` - Enable percentile normalization

- `-g, --gamma <FLOAT>` - Gamma correction [default: 1.0]
  - `< 1.0` = darker
  - `> 1.0` = brighter

- `-h, --help` - Print help
- `-V, --version` - Print version

### Examples

```bash
# Basic black & white with default kernel
image_filters input.jpg output.png

# Brighten with gamma
image_filters -g 1.5 input.jpg output.png

# Color dithering with normalization and gamma
image_filters -m color -n -g 1.2 input.jpg output.png

# Dominant mode with custom threshold
image_filters -m dominant -t 0.5 -n input.jpg output.png

# CMYK mode with Bayer kernel
image_filters -m cmyk -k bayer4 input.jpg output.png

# Use a discrete random seed
image_filters -k 3x3:1337 input.jpg output.png

# Use a continuous seed for smooth variation
image_filters -k 3x3c:42.5 input.jpg output.png

# Use an annealed kernel (optimized for max difference between neighbors)
image_filters -k anneal:4x4:12345 input.jpg output.png

# Use an annealed kernel with random seed
image_filters -k anneal:4x4 input.jpg output.png
```

## Kernel Seeds

### Discrete Seeds (3x3:SEED, 4x4:SEED)

Uses high avalanche - completely different patterns for different seeds. Good for random exploration.

```bash
image_filters -k 3x3:42 input.jpg output.png
image_filters -k 4x4:1337 input.jpg output.png
```

### Continuous Seeds (3x3c:SEED, 4x4c:SEED)

Uses low avalanche - similar seeds produce similar patterns. Good for optimization and fine-tuning.

```bash
image_filters -k 3x3c:42.5 input.jpg output.png
image_filters -k 3x3c:42.6 input.jpg output.png  # Very similar to 42.5
```

### Annealed Seeds (anneal:3x3[:SEED], anneal:4x4[:SEED])

Uses simulated annealing to find kernel arrangements that maximize differences between adjacent cells. This produces high-quality dithering patterns that avoid clustering artifacts.

- For 3x3: General simulated annealing optimization
- For 4x4: Uses constrained optimization (alternating rows pattern) - rows 1,3 contain high values (9-16), rows 2,4 contain low values (1-8)
- Same seed always produces the same kernel
- Omit seed to use current time (random)
- Score 373.02 is optimal for 4x4 kernels

```bash
# With specific seed (deterministic)
image_filters -k anneal:3x3:42 input.jpg output.png
image_filters -k anneal:4x4:12345 input.jpg output.png

# With random seed (different each time)
image_filters -k anneal:3x3 input.jpg output.png
image_filters -k anneal:4x4 input.jpg output.png
```

## Contact Sheet Generator

Explore different kernel seeds visually by generating a contact sheet showing the same image dithered with various seeds.

### Usage

```bash
cargo run --release --example contact_sheet -- <input> <output> [start_seed] [end_seed] [step] [cols]
```

### Arguments

- `<input>` - Input image path
- `<output>` - Output contact sheet image path
- `[start_seed]` - Starting seed value [default: 0.0]
- `[end_seed]` - Ending seed value [default: 20.0]
- `[step]` - Step between seeds [default: 1.0]
- `[cols]` - Number of columns in grid [default: 5]

### Examples

```bash
# Generate a 5-column grid testing seeds 0.0 through 20.0 (step 1.0)
cargo run --release --example contact_sheet -- input.jpg contact.png

# Test seeds 0.0 to 50.0 in steps of 2.5, arranged in 6 columns
cargo run --release --example contact_sheet -- input.jpg contact.png 0.0 50.0 2.5 6

# Fine-tune around seed 42: test 40.0 to 44.0 in steps of 0.2
cargo run --release --example contact_sheet -- input.jpg contact.png 40.0 44.0 0.2 5
```

The contact sheet displays each dithered version with its seed value labeled, making it easy to visually compare and find aesthetically pleasing kernels.

## License

MIT