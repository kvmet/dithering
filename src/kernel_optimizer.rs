/// Kernel optimizer for arranging values to maximize differences between adjacent cells
/// Uses simulated annealing with adaptive cooling and smart move strategies

use std::fmt;
use std::time::Instant;

/// Incremental scorer that caches pairwise scores for efficiency
/// When swapping two values, only recalculates affected pairs (O(n) instead of O(n²))
struct IncrementalScorer {
    size: usize,
    positions: Vec<(usize, usize)>,

    // Current total pairwise score (distance + alignment penalties)
    pairwise_score: f32,
}

impl IncrementalScorer {
    /// Initialize scorer with starting positions
    fn new(size: usize, positions: Vec<(usize, usize)>) -> Self {
        let mut scorer = IncrementalScorer {
            size,
            positions,
            pairwise_score: 0.0,
        };

        // Calculate initial score
        scorer.pairwise_score = scorer.calculate_full_pairwise_score();

        scorer
    }

    /// Get total score
    #[inline]
    fn total_score(&self) -> f32 {
        self.pairwise_score
    }

    /// Calculate pairwise score contribution for a single value index
    #[inline]
    fn calculate_pairwise_for_value(&self, val_idx: usize) -> f32 {
        let mut score = 0.0;
        let (r1, c1) = self.positions[val_idx];

        for other_idx in 0..self.positions.len() {
            if other_idx == val_idx {
                continue;
            }

            let (r2, c2) = self.positions[other_idx];

            // Calculate wrapped distance squared (toroidal)
            let dr = toroidal_distance_component(r1, r2, self.size);
            let dc = toroidal_distance_component(c1, c2, self.size);
            let distance_sq = (dr * dr + dc * dc) as f32;

            // Determine which index is smaller for consistent weighting
            let (i, j) = if val_idx < other_idx {
                (val_idx, other_idx)
            } else {
                (other_idx, val_idx)
            };

            // Weight by sequence distance (closer in sequence = more important)
            let sequence_distance = (j - i) as f32;
            let sequence_weight = 1.0 / sequence_distance;

            // Weight by position (early transitions more visible than late)
            let position_weight = 1.0 / ((i + 1) as f32).sqrt();

            let combined_weight = sequence_weight * position_weight;

            // Reward distance squared
            score += distance_sq * combined_weight;

            // Penalties for alignment (causes banding)
            let alignment_penalty = if r1 == r2 || c1 == c2 {
                ALIGNMENT_PENALTY_SAME_LINE * combined_weight
            } else if dr == dc {
                ALIGNMENT_PENALTY_DIAGONAL * combined_weight
            } else {
                0.0
            };

            score += alignment_penalty;
        }

        score
    }

    /// Calculate full pairwise score (used for initialization)
    fn calculate_full_pairwise_score(&self) -> f32 {
        let mut total = 0.0;

        for i in 0..self.positions.len() {
            for j in (i + 1)..self.positions.len() {
                let (r1, c1) = self.positions[i];
                let (r2, c2) = self.positions[j];

                let dr = toroidal_distance_component(r1, r2, self.size);
                let dc = toroidal_distance_component(c1, c2, self.size);
                let distance_sq = (dr * dr + dc * dc) as f32;

                let sequence_distance = (j - i) as f32;
                let sequence_weight = 1.0 / sequence_distance;
                let position_weight = 1.0 / ((i + 1) as f32).sqrt();
                let combined_weight = sequence_weight * position_weight;

                total += distance_sq * combined_weight;

                let alignment_penalty = if r1 == r2 || c1 == c2 {
                    ALIGNMENT_PENALTY_SAME_LINE * combined_weight
                } else if dr == dc {
                    ALIGNMENT_PENALTY_DIAGONAL * combined_weight
                } else {
                    0.0
                };

                total += alignment_penalty;
            }
        }

        total
    }

    /// Update score incrementally after swapping two value indices
    /// val_idx1 and val_idx2 are the VALUE indices (0-based, from the positions array)
    fn update_after_swap(&mut self, val_idx1: usize, val_idx2: usize) {
        // Subtract out old contributions from both values
        let old_contribution1 = self.calculate_pairwise_for_value(val_idx1);
        let old_contribution2 = self.calculate_pairwise_for_value(val_idx2);

        // Swap positions
        self.positions.swap(val_idx1, val_idx2);

        // Add back new contributions
        let new_contribution1 = self.calculate_pairwise_for_value(val_idx1);
        let new_contribution2 = self.calculate_pairwise_for_value(val_idx2);

        // Update pairwise score
        // Note: we subtract old_contribution/2 because each pair is counted twice
        // (once from each value's perspective), but we only want to count it once
        self.pairwise_score = self.pairwise_score
            - old_contribution1 / 2.0
            - old_contribution2 / 2.0
            + new_contribution1 / 2.0
            + new_contribution2 / 2.0;
    }

    /// Undo a swap (used when rejecting a move)
    fn undo_swap(&mut self, val_idx1: usize, val_idx2: usize) {
        // Just swap back and recalculate
        // We could be smarter here but swaps are cheap
        self.update_after_swap(val_idx1, val_idx2);
    }
}

// Scoring constants
const ALIGNMENT_PENALTY: f32 = 5.0; // Overall alignment penalty strength
const DIAGONAL_REJECTION: f32 = 1.5;
const ALIGNMENT_PENALTY_SAME_LINE: f32 = -ALIGNMENT_PENALTY; // Penalty for being in same row/column
const ALIGNMENT_PENALTY_DIAGONAL: f32 = DIAGONAL_REJECTION * 0.7071 * ALIGNMENT_PENALTY_SAME_LINE; // Penalty for being on a diagonal

/// Calculate toroidal (wrapped) distance component between two coordinates
#[inline]
fn toroidal_distance_component(a: usize, b: usize, size: usize) -> i32 {
    let diff = (a as i32 - b as i32).abs();
    diff.min(size as i32 - diff)
}

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
    /// - Toroidal (wrapped) squared distances between sequential values
    ///   (using squared distance gives quadratic reward, encouraging maximum spread)
    /// - Position-based weighting (early values weighted more)
    /// - Penalties for row/column/diagonal alignment
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

                // Calculate wrapped distance squared (toroidal)
                let dr = toroidal_distance_component(r1, r2, self.size);
                let dc = toroidal_distance_component(c1, c2, self.size);

                let distance_sq = (dr * dr + dc * dc) as f32;

                // Weight by sequence distance (closer in sequence = more important)
                let sequence_distance = (j - i) as f32;
                let sequence_weight = 1.0 / sequence_distance;

                // Weight by position (early transitions more visible than late)
                // Using i+1 to avoid division issues and because value indices are 0-based
                let position_weight = 1.0 / ((i + 1) as f32).sqrt();

                let combined_weight = sequence_weight * position_weight;

                // Reward distance squared (no sqrt needed, gives more weight to larger distances)
                // This is fine since we want to maximize separation
                total_score += distance_sq * combined_weight;

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

        total_score
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

    // Build initial positions and scorer
    let positions = {
        let kernel = Kernel::new(size, current.clone());
        kernel.build_positions()
    };

    // Initialize incremental scorer
    let mut scorer = IncrementalScorer::new(size, positions);
    let mut current_score = scorer.total_score();

    let mut best_kernel = Kernel::new(size, current.clone());
    let mut best_score = current_score;

    let mut temperature = initial_temp;
    let start_time = Instant::now();

    // Adaptive cooling parameters
    let mut accepts = 0;
    let mut rejects = 0;
    let check_interval = 10000;

    // Strategy: start with greedy/smart moves, transition to random
    let greedy_phase_end = iterations / 10; // First 10% is greedy-ish

    println!("Starting simulated annealing (incremental scoring + adaptive cooling + smart moves)...");
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
            if let Some((pos1, pos2)) = find_aligned_pair(&scorer.positions, size, &mut random) {
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

        // Get value indices for the positions we're swapping
        let val1 = current[i] as usize - 1;
        let val2 = current[j] as usize - 1;

        // Try swap in grid
        current.swap(i, j);

        // Update scorer incrementally
        scorer.update_after_swap(val1, val2);
        let new_score = scorer.total_score();

        let delta = new_score - current_score;

        // Accept if better, or sometimes if worse based on temperature
        let random_val = (random() as f64) / (u64::MAX as f64);
        if delta > 0.0 || random_val < (delta as f64 / temperature).exp() {
            // Accept the swap
            current_score = new_score;
            accepts += 1;

            if current_score > best_score {
                best_score = current_score;
                best_kernel = Kernel::new(size, current.clone());
            }
        } else {
            // Reject: swap back both current and scorer
            current.swap(i, j);
            scorer.undo_swap(val1, val2);
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
        let dr = toroidal_distance_component(r1, r2, size);
        let dc = toroidal_distance_component(c1, c2, size);

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
