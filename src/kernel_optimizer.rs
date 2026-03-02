/// Kernel optimizer for arranging values to maximize differences between adjacent cells
/// Uses simulated annealing with adaptive cooling and smart move strategies

use std::fmt;
use std::time::Instant;

/// Incremental scorer that caches pairwise scores for efficiency
/// When swapping two values, only recalculates affected pairs (O(n) instead of O(n²))
struct IncrementalScorer {
    size: usize,
    positions: Vec<(usize, usize)>,

    // Current total score with weights applied
    total_score: f32,

    // Store old score for efficient undo
    old_score: f32,

    // Precomputed score table: score_table[val_i][val_j] = score for that value pair
    // Indexed by value indices (0-based), stores score at their current positions
    score_table: Vec<Vec<f32>>,
}

impl IncrementalScorer {
    /// Initialize scorer with starting positions
    fn new(size: usize, positions: Vec<(usize, usize)>) -> Self {
        let n = positions.len();
        let mut scorer = IncrementalScorer {
            size,
            positions,
            total_score: 0.0,
            old_score: 0.0,
            score_table: vec![vec![0.0; n]; n],
        };

        // Build score lookup table
        scorer.build_score_table();

        // Calculate initial score from table
        scorer.total_score = scorer.calculate_full_score();

        scorer
    }

    /// Build the score lookup table for all value pairs at their current positions
    fn build_score_table(&mut self) {
        let n = self.positions.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let score = self.calculate_pair_score_direct(i, j);
                self.score_table[i][j] = score;
                self.score_table[j][i] = score; // Symmetric
            }
        }
    }

    /// Calculate score for a pair directly from positions (used for table building)
    fn calculate_pair_score_direct(&self, val_idx1: usize, val_idx2: usize) -> f32 {
        let (r1, c1) = self.positions[val_idx1];
        let (r2, c2) = self.positions[val_idx2];

        let dr = toroidal_distance_component(r1, r2, self.size);
        let dc = toroidal_distance_component(c1, c2, self.size);
        let distance_sq = (dr * dr + dc * dc) as f32;

        let (i, j) = if val_idx1 < val_idx2 {
            (val_idx1, val_idx2)
        } else {
            (val_idx2, val_idx1)
        };

        let sequence_distance = (j - i) as f32;
        let sequence_weight = 1.0 / sequence_distance;
        let position_weight = 1.0 / ((i + 1) as f32).sqrt();
        let combined_weight = sequence_weight * position_weight;
        let weighted_distance = distance_sq * combined_weight;

        // Branchless weight selection
        let is_vertical = (c1 == c2) as i32 as f32;
        let is_horizontal = (r1 == r2) as i32 as f32;
        let is_pos_diagonal = (dr == dc) as i32 as f32;
        let is_neg_diagonal = (dr == -dc) as i32 as f32;
        let is_knight = ((dr * dr + dc * dc) == 5) as i32 as f32;

        let weight = is_vertical * VERTICAL_WEIGHT
            + is_horizontal * HORIZONTAL_WEIGHT
            + is_pos_diagonal * POSITIVE_DIAGONAL_WEIGHT
            + is_neg_diagonal * NEGATIVE_DIAGONAL_WEIGHT
            + is_knight * KNIGHT_WEIGHT
            + (1.0 - is_vertical - is_horizontal - is_pos_diagonal - is_neg_diagonal - is_knight) * OTHER_WEIGHT;

        weight * weighted_distance
    }

    /// Get total score
    #[inline]
    fn total_score(&self) -> f32 {
        self.total_score
    }

    /// Get score for a pair from the lookup table
    #[inline]
    fn get_pair_score(&self, val_idx1: usize, val_idx2: usize) -> f32 {
        self.score_table[val_idx1][val_idx2]
    }

    /// Calculate full score from lookup table
    fn calculate_full_score(&self) -> f32 {
        let mut total = 0.0;
        let n = self.positions.len();

        for i in 0..n {
            for j in (i + 1)..n {
                total += self.score_table[i][j];
            }
        }

        total
    }

    /// Update score incrementally after swapping two value indices
    /// val_idx1 and val_idx2 are the VALUE indices (0-based, from the positions array)
    fn update_after_swap(&mut self, val_idx1: usize, val_idx2: usize) {
        // Save old score for potential undo
        self.old_score = self.total_score;

        // Calculate old pair scores from table (before swap)
        let mut old_pairs_sum = 0.0;
        let n = self.positions.len();

        for other_idx in 0..n {
            if other_idx == val_idx1 || other_idx == val_idx2 {
                continue;
            }
            old_pairs_sum += self.get_pair_score(val_idx1, other_idx);
            old_pairs_sum += self.get_pair_score(val_idx2, other_idx);
        }
        old_pairs_sum += self.get_pair_score(val_idx1, val_idx2);

        // Swap positions
        self.positions.swap(val_idx1, val_idx2);

        // Recalculate and update affected entries in score table
        for other_idx in 0..n {
            if other_idx == val_idx1 || other_idx == val_idx2 {
                continue;
            }

            // Recalculate scores for pairs involving the swapped values
            let score1 = self.calculate_pair_score_direct(val_idx1, other_idx);
            let score2 = self.calculate_pair_score_direct(val_idx2, other_idx);

            self.score_table[val_idx1][other_idx] = score1;
            self.score_table[other_idx][val_idx1] = score1;
            self.score_table[val_idx2][other_idx] = score2;
            self.score_table[other_idx][val_idx2] = score2;
        }

        // Update the pair between val_idx1 and val_idx2
        let score_between = self.calculate_pair_score_direct(val_idx1, val_idx2);
        self.score_table[val_idx1][val_idx2] = score_between;
        self.score_table[val_idx2][val_idx1] = score_between;

        // Calculate new pair scores from updated table
        let mut new_pairs_sum = 0.0;
        for other_idx in 0..n {
            if other_idx == val_idx1 || other_idx == val_idx2 {
                continue;
            }
            new_pairs_sum += self.get_pair_score(val_idx1, other_idx);
            new_pairs_sum += self.get_pair_score(val_idx2, other_idx);
        }
        new_pairs_sum += self.get_pair_score(val_idx1, val_idx2);

        // Update total score
        self.total_score = self.total_score - old_pairs_sum + new_pairs_sum;
    }

    /// Undo a swap (used when rejecting a move)
    fn undo_swap(&mut self, val_idx1: usize, val_idx2: usize) {
        // Swap positions back
        self.positions.swap(val_idx1, val_idx2);

        // Restore old score (much faster than recalculating)
        self.total_score = self.old_score;
    }
}

// Scoring weights for each geometric bucket
// Lower weight = penalize this geometry (we don't want distance here)
// Higher weight = reward this geometry (we want distance here)
const VERTICAL_WEIGHT: f32 = 0.2;            // Weight for vertical (same column) alignments - BAD
const HORIZONTAL_WEIGHT: f32 = 0.2;          // Weight for horizontal (same row) alignments - BAD
const POSITIVE_DIAGONAL_WEIGHT: f32 = 0.5;   // Weight for positive diagonal (slope = 1) - LESS BAD
const NEGATIVE_DIAGONAL_WEIGHT: f32 = 0.5;   // Weight for negative diagonal (slope = -1) - LESS BAD
const KNIGHT_WEIGHT: f32 = 1.0;              // Weight for knight move patterns (2,1 or 1,2) - MEDIUM
const OTHER_WEIGHT: f32 = 2.0;               // Weight for all other geometric relationships - GOOD

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

                // Distance contribution weighted by combined weight
                let weighted_distance = distance_sq * combined_weight;

                // Classify by geometric relationship and add to total with appropriate weight
                if c1 == c2 {
                    total_score += VERTICAL_WEIGHT * weighted_distance;
                } else if r1 == r2 {
                    total_score += HORIZONTAL_WEIGHT * weighted_distance;
                } else if dr == dc {
                    total_score += POSITIVE_DIAGONAL_WEIGHT * weighted_distance;
                } else if dr == -dc {
                    total_score += NEGATIVE_DIAGONAL_WEIGHT * weighted_distance;
                } else {
                    total_score += OTHER_WEIGHT * weighted_distance;
                }
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
