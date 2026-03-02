/// Kernel optimizer for arranging values to maximize differences between adjacent cells
/// Uses simulated annealing with adaptive cooling and smart move strategies

use std::fmt;
use std::time::Instant;

// Scoring constants
const ALIGNMENT_PENALTY_SAME_LINE: f32 = -5.0;
const ALIGNMENT_PENALTY_DIAGONAL: f32 = -3.0;
const CLUSTERING_PENALTY_MULTIPLIER: f32 = 2.0;
const NEARBY_SEQUENTIAL_WINDOW: usize = 3;

#[derive(Clone)]
pub struct Kernel {
    pub grid: Vec<f32>,
    pub size: usize,
}

impl Kernel {
    pub fn new(size: usize, values: Vec<f32>) -> Self {
        assert_eq!(values.len(), size * size, "Values must match grid size");
        Kernel { grid: values, size }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.grid[row * self.size + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.grid[row * self.size + col] = value;
    }

    /// Build position map: value -> (row, col)
    fn build_positions(&self) -> Vec<(usize, usize)> {
        let mut positions = vec![(0, 0); self.grid.len()];
        for row in 0..self.size {
            for col in 0..self.size {
                let value = self.get(row, col);
                let index = (value as usize) - 1;
                positions[index] = (row, col);
            }
        }
        positions
    }

    /// Calculate the score for this arrangement
    /// Higher score = better for ordered dithering
    ///
    /// Scoring considers:
    /// - Toroidal (wrapped) distances between sequential values
    /// - Position-based weighting (early values weighted more)
    /// - Penalties for row/column/diagonal alignment
    /// - Clustering detection within appropriate radius
    pub fn score(&self) -> f32 {
        let positions = self.build_positions();
        self.score_with_positions(&positions)
    }

    /// Calculate score using pre-built position map
    /// This is more efficient when positions are already known
    fn score_with_positions(&self, positions: &[(usize, usize)]) -> f32 {
        let mut total_score = 0.0;

        // Calculate pairwise distances with weighting
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let (r1, c1) = positions[i];
                let (r2, c2) = positions[j];

                // Calculate wrapped distance (toroidal)
                let dr = ((r1 as i32 - r2 as i32).abs()).min(self.size as i32 - (r1 as i32 - r2 as i32).abs());
                let dc = ((c1 as i32 - c2 as i32).abs()).min(self.size as i32 - (c1 as i32 - c2 as i32).abs());

                let distance = ((dr * dr + dc * dc) as f32).sqrt();

                // Weight by sequence distance (closer in sequence = more important)
                let sequence_distance = (j - i) as f32;
                let sequence_weight = 1.0 / sequence_distance;

                // Weight by position (early transitions more visible than late)
                // Using i+1 to avoid division issues and because value indices are 0-based
                let position_weight = 1.0 / ((i + 1) as f32).sqrt();

                let combined_weight = sequence_weight * position_weight;

                // Reward distance, weighted by importance
                total_score += distance * combined_weight;

                // Penalties for alignment (causes banding)
                let alignment_penalty = if r1 == r2 || c1 == c2 {
                    // Same row or column - very bad
                    ALIGNMENT_PENALTY_SAME_LINE * combined_weight
                } else if dr == dc {
                    // Diagonal alignment - bad but less so
                    ALIGNMENT_PENALTY_DIAGONAL * combined_weight
                } else {
                    0.0
                };

                total_score += alignment_penalty;
            }
        }

        // Clustering detection
        // Radius based on grid size: size - 3 (so 4x4 checks immediate neighbors, 5x5 checks 2 levels, etc.)
        let cluster_radius = if self.size > 3 { self.size - 3 } else { 0 };

        if cluster_radius > 0 {
            let cluster_penalty = self.calculate_clustering_penalty(&positions, cluster_radius);
            total_score += cluster_penalty;
        }

        total_score
    }

    /// Calculate penalty for sequential values being clustered together
    fn calculate_clustering_penalty(&self, positions: &[(usize, usize)], radius: usize) -> f32 {
        let mut penalty = 0.0;
        let radius_sq = (radius * radius) as i32;

        for i in 0..positions.len() {
            let (r1, c1) = positions[i];
            let mut nearby_sequential = 0;

            // Check for sequential values within radius
            // Look in window [i-NEARBY_SEQUENTIAL_WINDOW, i+NEARBY_SEQUENTIAL_WINDOW] excluding i itself
            let start = if i >= NEARBY_SEQUENTIAL_WINDOW { i - NEARBY_SEQUENTIAL_WINDOW } else { 0 };
            let end = (i + NEARBY_SEQUENTIAL_WINDOW + 1).min(positions.len());

            for j in start..end {
                if j == i {
                    continue;
                }

                let (r2, c2) = positions[j];

                // Calculate toroidal distance squared
                let dr = ((r1 as i32 - r2 as i32).abs()).min(self.size as i32 - (r1 as i32 - r2 as i32).abs());
                let dc = ((c1 as i32 - c2 as i32).abs()).min(self.size as i32 - (c1 as i32 - c2 as i32).abs());
                let dist_sq = dr * dr + dc * dc;

                if dist_sq <= radius_sq {
                    nearby_sequential += 1;
                }
            }

            // Penalize based on how many sequential neighbors are nearby
            // Weight by position (early values matter more)
            let position_weight = 1.0 / ((i + 1) as f32).sqrt();
            penalty -= (nearby_sequential as f32) * CLUSTERING_PENALTY_MULTIPLIER * position_weight;
        }

        penalty
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.size {
            for col in 0..self.size {
                write!(f, "{:6.2} ", self.get(row, col))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Optimize kernel using simulated annealing with adaptive cooling and smart move strategies
pub fn optimize_kernel(
    size: usize,
    values: Vec<f32>,
    iterations: usize,
    initial_temp: f64,
    cooling_rate: f64,
    seed: u64,
) -> Kernel {
    assert_eq!(values.len(), size * size, "Values must match grid size");

    // Simple pseudo-random number generator (LCG)
    let mut rng_state = seed;
    let mut random = || {
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        rng_state
    };

    // Start with random arrangement
    let mut current = values.clone();
    for i in (1..current.len()).rev() {
        let j = (random() as usize) % (i + 1);
        current.swap(i, j);
    }

    let mut current_kernel = Kernel::new(size, current.clone());
    let mut current_score = current_kernel.score();
    let mut positions = current_kernel.build_positions();

    let mut best_kernel = current_kernel.clone();
    let mut best_score = current_score;

    let mut temperature = initial_temp;
    let start_time = Instant::now();

    // Adaptive cooling parameters
    let mut accepts = 0;
    let mut rejects = 0;
    let check_interval = 10000;

    // Strategy: start with greedy/smart moves, transition to random
    let greedy_phase_end = iterations / 10; // First 10% is greedy-ish

    println!("Starting simulated annealing (adaptive cooling + smart moves)...");
    println!("Initial score: {:.2}", current_score);

    for iteration in 0..iterations {
        // Progress reporting
        if iteration > 0 && (iteration % 1_000_000 == 0 || iteration == iterations - 1) {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = iteration as f64 / elapsed;
            let accept_rate = if accepts + rejects > 0 {
                accepts as f64 / (accepts + rejects) as f64
            } else {
                0.0
            };
            println!(
                "Iteration {}/{} ({:.0} iter/s) - Best: {:.2}, Current: {:.2}, Temp: {:.4}, Accept: {:.1}%",
                iteration, iterations, rate, best_score, current_score, temperature, accept_rate * 100.0
            );
        }

        // Adaptive cooling: adjust temperature directly based on acceptance rate
        if iteration > 0 && iteration % check_interval == 0 {
            let accept_rate = accepts as f64 / (accepts + rejects) as f64;
            if accept_rate > 0.6 {
                // Accepting too much, cool down faster
                temperature *= 0.9;
            } else if accept_rate < 0.05 {
                // Barely accepting anything, warm up a bit
                temperature *= 1.1;
            }
            // Otherwise keep normal cooling rate
            accepts = 0;
            rejects = 0;
        }

        // Pick two random positions to swap
        let i;
        let j;

        // Smart move strategy: during greedy phase, try to fix alignments
        if iteration < greedy_phase_end && (random() % 3) == 0 {
            // Try to swap aligned values
            if let Some((pos1, pos2)) = find_aligned_pair(&positions, size, &mut random) {
                i = pos1;
                j = pos2;
            } else {
                i = (random() as usize) % current.len();
                j = (random() as usize) % current.len();
            }
        } else {
            // Random swap
            i = (random() as usize) % current.len();
            j = (random() as usize) % current.len();
        }

        if i == j {
            continue;
        }

        // Try swap
        current.swap(i, j);
        let new_kernel = Kernel::new(size, current.clone());

        // Update positions map for the swap
        let val1 = current[i] as usize - 1;
        let val2 = current[j] as usize - 1;
        positions.swap(val1, val2);

        let new_score = new_kernel.score_with_positions(&positions);

        let delta = new_score - current_score;

        // Accept if better, or sometimes if worse based on temperature
        let random_val = (random() as f64) / (u64::MAX as f64);
        if delta > 0.0 || random_val < (delta as f64 / temperature).exp() {
            // Accept the swap
            current_score = new_score;
            current_kernel = new_kernel;
            // Positions already updated above
            accepts += 1;

            if current_score > best_score {
                best_score = current_score;
                best_kernel = current_kernel.clone();
            }
        } else {
            // Reject: swap back both current and positions
            current.swap(i, j);
            let val1 = current[i] as usize - 1;
            let val2 = current[j] as usize - 1;
            positions.swap(val1, val2);
            rejects += 1;
        }

        // Cool down
        temperature *= cooling_rate;
    }

    println!("\nSimulated annealing complete!");
    println!("Best score found: {:.2}", best_score);
    println!("Total time: {:.2}s", start_time.elapsed().as_secs_f64());

    best_kernel
}

/// Find a pair of values that are currently aligned (for smart swapping)
fn find_aligned_pair(
    positions: &[(usize, usize)],
    size: usize,
    random: &mut impl FnMut() -> u64,
) -> Option<(usize, usize)> {
    // Try a few random pairs to find aligned ones
    for _ in 0..10 {
        let i = (random() as usize) % positions.len();
        let j = (random() as usize) % positions.len();

        if i == j {
            continue;
        }

        let (r1, c1) = positions[i];
        let (r2, c2) = positions[j];

        // Check if aligned (row, column, or diagonal)
        let dr = ((r1 as i32 - r2 as i32).abs()).min(size as i32 - (r1 as i32 - r2 as i32).abs());
        let dc = ((c1 as i32 - c2 as i32).abs()).min(size as i32 - (c1 as i32 - c2 as i32).abs());

        if r1 == r2 || c1 == c2 || dr == dc {
            // Found aligned pair - return grid positions
            let pos1 = r1 * size + c1;
            let pos2 = r2 * size + c2;
            return Some((pos1, pos2));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2x2_scoring() {
        let kernel1 = Kernel::new(2, vec![1.0, 2.0, 3.0, 4.0]);
        let kernel2 = Kernel::new(2, vec![1.0, 4.0, 3.0, 2.0]);

        let score1 = kernel1.score();
        let score2 = kernel2.score();

        println!("Kernel 1 score: {:.2}", score1);
        println!("Kernel 2 score: {:.2}", score2);

        assert!(score2 > score1, "Kernel 2 should score higher");
    }

    #[test]
    fn test_4x4_optimization() {
        let values: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let kernel = optimize_kernel(4, values, 100_000, 10.0, 0.99999, 12345);

        println!("Optimized 4x4 arrangement:");
        println!("{}", kernel);
        println!("Score: {:.2}", kernel.score());
    }
}
