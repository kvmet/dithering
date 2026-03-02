/// Kernel expander for CMYK dithering
/// Takes a root kernel and creates 4 channel-specific kernels (C, M, Y, K)
/// with minimal spatial overlap to reduce artifacts



/// Represents a set of CMYK kernels derived from a root kernel
pub struct CmykKernels {
    pub cyan: Vec<f64>,
    pub magenta: Vec<f64>,
    pub yellow: Vec<f64>,
    pub black: Vec<f64>,
    pub size: usize,
}

impl CmykKernels {
    /// Get the kernel for a specific channel
    pub fn get_channel(&self, channel: usize) -> &[f64] {
        match channel {
            0 => &self.cyan,
            1 => &self.magenta,
            2 => &self.yellow,
            3 => &self.black,
            _ => panic!("Invalid channel index: {}", channel),
        }
    }
}

/// Expand a root kernel into 4 CMYK kernels with minimal overlap
///
/// Strategy:
/// - Each kernel gets every 4th threshold value
/// - Rotate which values each channel gets to minimize spatial overlap
/// - Use spatial distribution to ensure different channels activate at different positions
pub fn expand_kernel_cmyk(root_kernel: &[f64], size: usize) -> CmykKernels {
    assert_eq!(root_kernel.len(), size * size, "Root kernel must match size");

    let n = root_kernel.len();

    // Build index map: for each threshold value, where is it in the grid?
    let mut value_positions: Vec<(f64, usize)> = root_kernel
        .iter()
        .enumerate()
        .map(|(idx, &val)| (val, idx))
        .collect();

    // Sort by threshold value
    value_positions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Initialize output kernels with maximum values (so unused positions never activate)
    let mut cyan: Vec<f64> = vec![1.0; n];
    let mut magenta: Vec<f64> = vec![1.0; n];
    let mut yellow: Vec<f64> = vec![1.0; n];
    let mut black: Vec<f64> = vec![1.0; n];

    // Distribute values among channels
    // Each channel gets every 4th value, but offset by rotation
    // This ensures temporal distribution (values activate at different times)
    // Combined with spatial distribution from optimizer, this minimizes overlap

    for (rank, &(_value, position)) in value_positions.iter().enumerate() {
        let channel = rank % 4;
        let channel_rank = rank / 4;
        let channel_normalized = (channel_rank as f64) / ((n / 4) as f64 + 0.5);

        match channel {
            0 => cyan[position] = channel_normalized,
            1 => magenta[position] = channel_normalized,
            2 => yellow[position] = channel_normalized,
            3 => black[position] = channel_normalized,
            _ => unreachable!(),
        }
    }

    CmykKernels {
        cyan,
        magenta,
        yellow,
        black,
        size,
    }
}

/// Expand kernel with rotation strategy to minimize spatial clustering
/// This version rotates the assignment pattern to spread out activations
pub fn expand_kernel_cmyk_rotated(root_kernel: &[f64], size: usize) -> CmykKernels {
    assert_eq!(root_kernel.len(), size * size, "Root kernel must match size");

    let n = root_kernel.len();

    // Build sorted list of (value, row, col)
    let mut value_positions: Vec<(f64, usize, usize)> = Vec::new();
    for row in 0..size {
        for col in 0..size {
            let idx = row * size + col;
            value_positions.push((root_kernel[idx], row, col));
        }
    }
    value_positions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Initialize output kernels
    let mut cyan: Vec<f64> = vec![1.0; n];
    let mut magenta: Vec<f64> = vec![1.0; n];
    let mut yellow: Vec<f64> = vec![1.0; n];
    let mut black: Vec<f64> = vec![1.0; n];

    // Assign based on position in grid to add spatial diversity
    for (rank, &(_value, row, col)) in value_positions.iter().enumerate() {
        let position = row * size + col;

        // Use position parity to rotate channel assignment
        // This breaks up spatial patterns further
        let spatial_offset = (row + col) % 4;
        let channel = (rank + spatial_offset) % 4;

        let channel_rank = rank / 4;
        let channel_normalized = (channel_rank as f64) / ((n / 4) as f64 + 0.5);

        match channel {
            0 => cyan[position] = channel_normalized,
            1 => magenta[position] = channel_normalized,
            2 => yellow[position] = channel_normalized,
            3 => black[position] = channel_normalized,
            _ => unreachable!(),
        }
    }

    CmykKernels {
        cyan,
        magenta,
        yellow,
        black,
        size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_expansion() {
        // Simple 2x2 kernel
        let root = vec![0.0, 0.33, 0.66, 1.0];
        let cmyk = expand_kernel_cmyk(&root, 2);

        assert_eq!(cmyk.cyan.len(), 4);
        assert_eq!(cmyk.magenta.len(), 4);
        assert_eq!(cmyk.yellow.len(), 4);
        assert_eq!(cmyk.black.len(), 4);

        // Each channel should have exactly one low value
        for channel in [&cmyk.cyan, &cmyk.magenta, &cmyk.yellow, &cmyk.black] {
            let active_count = channel.iter().filter(|&&v| v < 0.5).count();
            assert_eq!(active_count, 1, "Each channel should activate exactly one position early");
        }
    }


}
