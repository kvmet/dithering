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
  - `posterize` - Posterized CMY with spread/bloom and dithered black

- `-k, --kernel <KERNEL>` - Dither kernel [default: anneal:4x4]
  - `bayer2` - Bayer 2x2 pattern
  - `bayer4` - Bayer 4x4 pattern
  - `anneal:4x4[:SEED]` or `anneal:5x5[:SEED]` - Optimized via simulated annealing (e.g., `anneal:4x4:12345` or `anneal:4x4` for random)

- `-t, --threshold <FLOAT>` - Tie threshold for dominant mode [default: 0.3]
  - Range: 0.0-1.0
  - Only affects `dominant` mode

- `-n, --normalize` - Enable percentile normalization

- `-g, --gamma <FLOAT>` - Gamma correction (auto-computed if not specified)
  - `< 1.0` = darker
  - `> 1.0` = brighter
  - If not specified, automatically computed to match input/output brightness

- `--auto-gamma-offset <FLOAT>` - Multiplier for auto-computed gamma [default: 0.8]
  - Compensates for sRGB→linear darkening

- `-W, --width <PIXELS>` - Target width in pixels
  - Scales height proportionally to maintain aspect ratio if height not specified

- `-H, --height <PIXELS>` - Target height in pixels
  - Scales width proportionally to maintain aspect ratio if width not specified
  - If both width and height specified, image is stretched to exact dimensions

- `--print-kernel` - Print kernel values and exit
  - Use with `-k` to specify which kernel to print

- `--print-scores` - Print score table and exit
  - Use with `-k` to specify which kernel to analyze

- `--spread-radius <PIXELS>` - Spread radius for posterize mode [default: 3]
  - Grows/blooms CMY color areas in posterize mode
  - Higher values create larger color bleeds from colored regions

- `-h, --help` - Print help
- `-V, --version` - Print version

### Examples

```bash
# Basic black & white with default kernel (auto-gamma)
image_filters input.jpg output.png

# Resize to 800px wide (maintaining aspect ratio)
image_filters -W 800 input.jpg output.png

# Resize to specific dimensions (stretches if aspect ratio differs)
image_filters -W 800 -H 600 input.jpg output.png

# Brighten with manual gamma
image_filters -g 1.5 input.jpg output.png

# Color dithering with normalization and manual gamma
image_filters -m color -n -g 1.2 input.jpg output.png

# Dominant mode with custom threshold
image_filters -m dominant -t 0.5 -n input.jpg output.png

# Posterize mode with CMY spread and dithered blacks
image_filters -m posterize input.jpg output.png

# Posterize with more spread for larger color bleeds
image_filters -m posterize --spread-radius 5 input.jpg output.png

# Use an annealed kernel (optimized for max difference between neighbors)
image_filters -k anneal:4x4:12345 input.jpg output.png

# Use an annealed kernel with random seed
image_filters -k anneal:4x4 input.jpg output.png

# Use a 5x5 annealed kernel
image_filters -k anneal:5x5:abc123 input.jpg output.png

# Print kernel values and their quality score
image_filters --print-kernel -k anneal:4x4 input.jpg output.png

# Adjust auto-gamma compensation (default is 0.8)
image_filters --auto-gamma-offset 1.0 input.jpg output.png
```

## Annealed Kernels

### Annealed Seeds (anneal:4x4[:SEED], anneal:5x5[:SEED])

Uses simulated annealing to find kernel arrangements that maximize differences between adjacent cells. This produces high-quality dithering patterns that avoid clustering artifacts.

```bash
# With specific seed (deterministic)
image_filters -k anneal:4x4:12345 input.jpg output.png
image_filters -k anneal:5x5:abc123 input.jpg output.png

# With random seed (different each time)
image_filters -k anneal:4x4 input.jpg output.png
image_filters -k anneal:5x5 input.jpg output.png
```



## License

MIT
