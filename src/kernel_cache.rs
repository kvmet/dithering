/// Kernel caching system using TOML files
///
/// Kernels are stored in a `kernels/` directory in the project root.
/// Each file contains size, seed, and the computed grid values.
/// The filename can be anything - the TOML contents are the source of truth.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use crate::kernel_optimizer::Kernel;

const KERNELS_DIR: &str = "kernels";

#[derive(Debug, Serialize, Deserialize)]
struct KernelFile {
    size: usize,
    seed: u64,
    grid: Vec<f64>,
}

/// Try to load a cached kernel matching the given size and seed
///
/// Searches all .toml files in the kernels/ directory for a match.
/// Returns None if no matching kernel is found.
pub fn load_kernel(size: usize, seed: u64) -> Option<Kernel> {
    let kernels_path = Path::new(KERNELS_DIR);

    if !kernels_path.exists() {
        return None;
    }

    // Read all .toml files in the kernels directory
    let entries = fs::read_dir(kernels_path).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) != Some("toml") {
            continue;
        }

        // Try to read and parse the file
        if let Ok(contents) = fs::read_to_string(&path) {
            if let Ok(kernel_file) = toml::from_str::<KernelFile>(&contents) {
                // Check if this matches our size and seed
                if kernel_file.size == size && kernel_file.seed == seed {
                    // Validate the grid size matches
                    if kernel_file.grid.len() == size * size {
                        return Some(Kernel::new(size, kernel_file.grid));
                    }
                }
            }
        }
    }

    None
}

/// Save a kernel to a TOML file in the kernels/ directory
///
/// The filename will be `kernel_{size}x{size}_seed{seed}.toml` by default.
/// Creates the kernels/ directory if it doesn't exist.
pub fn save_kernel(kernel: &Kernel, seed: u64) -> Result<PathBuf, std::io::Error> {
    let kernels_path = Path::new(KERNELS_DIR);

    // Create kernels directory if it doesn't exist
    if !kernels_path.exists() {
        fs::create_dir(kernels_path)?;
    }

    let kernel_file = KernelFile {
        size: kernel.size,
        seed,
        grid: kernel.grid.clone(),
    };

    let toml_string = toml::to_string_pretty(&kernel_file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let filename = format!("kernel_{}x{}_seed{}.toml", kernel.size, kernel.size, seed);
    let filepath = kernels_path.join(filename);

    fs::write(&filepath, toml_string)?;

    Ok(filepath)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_serialization() {
        let kernel_file = KernelFile {
            size: 2,
            seed: 12345,
            grid: vec![1.0, 2.0, 3.0, 4.0],
        };

        let toml_string = toml::to_string_pretty(&kernel_file).unwrap();
        let parsed: KernelFile = toml::from_str(&toml_string).unwrap();

        assert_eq!(parsed.size, 2);
        assert_eq!(parsed.seed, 12345);
        assert_eq!(parsed.grid, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
