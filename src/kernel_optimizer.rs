/// Kernel optimizer for arranging values to maximize differences between adjacent cells
/// Uses simulated annealing with adaptive cooling and smart move strategies

use std::fmt;
use std::time::Instant;

// Scoring weights for each geometric bucket
// Lower weight = penalize this geometry (we don't want distance here)
// Higher weight = reward this geometry (we want distance here)
const VERTICAL_WEIGHT: f64 = 0.2;            // Weight for vertical (same column) alignments - BAD
const HORIZONTAL_WEIGHT: f64 = 0.2;          // Weight for horizontal (same row) alignments - BAD
const POSITIVE_DIAGONAL_WEIGHT: f64 = 0.3;   // Weight for positive diagonal (slope = 1) - LESS BAD
const NEGATIVE_DIAGONAL_WEIGHT: f64 = 0.3;   // Weight for negative diagonal (slope = -1) - LESS BAD
const KNIGHT_WEIGHT: f64 = 0.4;              // Weight for knight move patterns (2,1 or 1,2) - MEDIUM
const OTHER_WEIGHT: f64 = 1.0;               // Weight for all other geometric relationships - GOOD

// Linear Congruential Generator (LCG) constants
const LCG_MULTIPLIER: u64 = 1664525;
const LCG_INCREMENT: u64 = 1013904223;

// Annealing calibration constants
const CALIBRATION_SAMPLES: usize = 1000;
const CALIBRATION_TEMP_MULTIPLIER: f64 = 2.0; // initial_temp = avg_delta * this
const TARGET_FINAL_TEMP_RATIO: f64 = 0.01; // Cool to 1% of initial temp

// Annealing reporting
const REPORT_INTERVAL: usize = 1_000_000;

// Sequence weight interpolation
const NO_SEQUENCE_PENALTY: f64 = 1.0; // Weight when sequence_weight_strength = 0

/// Static score lookup table precomputed once for all possible position pairs
/// Stores only geometry_weight * distance_sq (independent of value assignments)
/// Position and sequence weights are applied at lookup time
pub struct ScoreLookup {
    // Flattened 2D array: [pos_i * n + pos_j]
    table: Vec<f64>,
    n: usize, // grid size (total number of values/positions)
    size: usize, // grid width/height
    // Precomputed weights for fast lookup
    sequence_weights: Vec<f64>, // [value_distance] -> weight
    position_weights: Vec<f64>, // [smaller_val] -> weight
}

impl ScoreLookup {
    pub fn new(size: usize) -> Self {
        Self::new_with_sequence_weight(size, 1.0)
    }

    pub fn new_with_sequence_weight(size: usize, sequence_weight_strength: f64) -> Self {
        let n = size * size;
        let table_size = n * n;
        let mut table = vec![0.0; table_size];

        // Precompute sequence weights for all possible value distances
        let mut sequence_weights = vec![NO_SEQUENCE_PENALTY; n];
        if sequence_weight_strength > 0.0 {
            for value_distance in 1..n {
                let raw_weight = 1.0 / value_distance as f64;
                sequence_weights[value_distance] = NO_SEQUENCE_PENALTY + sequence_weight_strength * (raw_weight - NO_SEQUENCE_PENALTY);
            }
        }

        // Precompute position weights for all possible smaller values
        let mut position_weights = vec![1.0; n];
        for smaller_val in 0..n {
            position_weights[smaller_val] = 1.0 / (smaller_val + 1) as f64;
        }

        // Precompute geometry_weight * distance_sq for all position pairs
        for pos_i in 0..n {
            let (r1, c1) = (pos_i / size, pos_i % size);

            for pos_j in 0..n {
                if pos_i == pos_j {
                    continue;
                }

                let (r2, c2) = (pos_j / size, pos_j % size);

                let dr = toroidal_distance_component(r1, r2, size);
                let dc = toroidal_distance_component(c1, c2, size);
                let distance_sq = (dr * dr + dc * dc) as f64;

                // Branchless geometry weight selection
                let is_vertical = (c1 == c2) as i32 as f64;
                let is_horizontal = (r1 == r2) as i32 as f64;
                let is_pos_diagonal = (dr == dc) as i32 as f64;
                let is_neg_diagonal = (dr == -dc) as i32 as f64;
                let is_knight = ((dr * dr + dc * dc) == 5) as i32 as f64;

                let geometry_weight = is_vertical * VERTICAL_WEIGHT
                    + is_horizontal * HORIZONTAL_WEIGHT
                    + is_pos_diagonal * POSITIVE_DIAGONAL_WEIGHT
                    + is_neg_diagonal * NEGATIVE_DIAGONAL_WEIGHT
                    + is_knight * KNIGHT_WEIGHT
                    + (1.0 - is_vertical - is_horizontal - is_pos_diagonal - is_neg_diagonal - is_knight) * OTHER_WEIGHT;

                // Store only the geometry-weighted distance score
                let base_score = geometry_weight * distance_sq;
                let idx = pos_i * n + pos_j;
                table[idx] = base_score;
            }
        }

        ScoreLookup { table, n, size, sequence_weights, position_weights }
    }

    #[inline]
    fn get(&self, val_i: usize, val_j: usize, pos_i: usize, pos_j: usize) -> f64 {
        // Get base geometry-weighted distance score (unchecked for speed)
        // Note: when pos_i == pos_j, table entry is 0.0, so result is correctly 0.0
        let idx = pos_i * self.n + pos_j;
        // SAFETY: pos_i and pos_j are derived from positions array which has length n,
        // so idx = pos_i * n + pos_j is always < n * n = table.len()
        let base_score = unsafe { *self.table.get_unchecked(idx) };

        // Apply precomputed value-dependent weights
        let value_distance = if val_i > val_j { val_i - val_j } else { val_j - val_i };
        // SAFETY: value_distance is the absolute difference of two indices < n,
        // so value_distance < n = sequence_weights.len()
        let sequence_weight = unsafe { *self.sequence_weights.get_unchecked(value_distance) };

        let smaller_val = val_i.min(val_j);
        // SAFETY: smaller_val is min of two value indices < n,
        // so smaller_val < n = position_weights.len()
        let position_weight = unsafe { *self.position_weights.get_unchecked(smaller_val) };

        base_score * sequence_weight * position_weight
    }

    /// Print the score matrix for a given value distance
    /// Shows how scores would be calculated if values at distance `value_distance` were placed at each position pair
    pub fn print_score_matrix(&self, value_distance: usize, val_i: usize, val_j: usize) {
        if value_distance == 0 || value_distance >= self.n {
            println!("Invalid value_distance: {} (must be 1..{})", value_distance, self.n - 1);
            return;
        }

        println!("Score matrix for values {} and {} (distance={}, {}x{} grid):", val_i, val_j, value_distance, self.size, self.size);
        println!("Each cell shows the score contribution for that position pair.");
        println!();

        // Print header with column coordinates
        print!("      ");
        for col in 0..self.n {
            print!("{:7}", format!("({})", col));
        }
        println!();

        // Print separator
        print!("      ");
        for _ in 0..self.n {
            print!("-------");
        }
        println!();

        // Print each row
        for pos_i in 0..self.n {
            print!("({:2}) |", pos_i);
            for pos_j in 0..self.n {
                if pos_i == pos_j {
                    print!("   -   ");
                } else {
                    let score = self.get(val_i, val_j, pos_i, pos_j);
                    print!("{:7.3}", score);
                }
            }
            println!();
        }
        println!();
    }
}

/// Incremental scorer that uses static lookup table for O(1) pair score lookups
pub struct IncrementalScorer {
    positions: Vec<usize>, // positions[val_idx] = flat position index where that value is located
    lookup: ScoreLookup,

    // Current total score with weights applied
    total_score: f64,

    // Store old score for efficient undo
    old_score: f64,
}

impl IncrementalScorer {
    /// Initialize scorer with starting positions
    pub fn new(size: usize, positions: Vec<usize>) -> Self {
        Self::new_with_sequence_weight(size, positions, 1.0)
    }

    pub fn new_with_sequence_weight(size: usize, positions: Vec<usize>, sequence_weight_strength: f64) -> Self {
        println!("Building static score lookup table (sequence_weight_strength={:.2})...", sequence_weight_strength);
        let lookup = ScoreLookup::new_with_sequence_weight(size, sequence_weight_strength);
        println!("Lookup table built: {} entries ({:.2} MB)",
                 lookup.table.len(),
                 (lookup.table.len() * std::mem::size_of::<f64>()) as f64 / 1024.0 / 1024.0);

        let mut scorer = IncrementalScorer {
            positions,
            lookup,
            total_score: 0.0,
            old_score: 0.0,
        };

        // Calculate initial score from lookup table
        scorer.total_score = scorer.calculate_full_score();

        scorer
    }

    /// Get total score
    #[inline]
    pub fn total_score(&self) -> f64 {
        self.total_score
    }

    /// Get score for a pair from the lookup table
    #[inline]
    fn get_pair_score(&self, val_idx1: usize, val_idx2: usize) -> f64 {
        let pos_i = self.positions[val_idx1];
        let pos_j = self.positions[val_idx2];
        self.lookup.get(val_idx1, val_idx2, pos_i, pos_j)
    }

    /// Calculate full score from lookup table
    fn calculate_full_score(&self) -> f64 {
        let mut total = 0.0;
        let n = self.positions.len();

        for i in 0..n {
            for j in (i + 1)..n {
                total += self.get_pair_score(i, j);
            }
        }

        total
    }

    /// Update score incrementally after swapping two value indices
    /// val_idx1 and val_idx2 are the VALUE indices (0-based, from the positions array)
    fn update_after_swap(&mut self, val_idx1: usize, val_idx2: usize) {
        // Save old score for potential undo
        self.old_score = self.total_score;

        // Calculate score delta in single pass
        let mut delta = 0.0;
        let n = self.positions.len();

        // Calculate old scores for val_idx1 and val_idx2 pairs
        for other_idx in 0..n {
            if other_idx != val_idx1 && other_idx != val_idx2 {
                delta -= self.get_pair_score(val_idx1, other_idx);
                delta -= self.get_pair_score(val_idx2, other_idx);
            }
        }
        delta -= self.get_pair_score(val_idx1, val_idx2);

        // Swap positions
        self.positions.swap(val_idx1, val_idx2);

        // Calculate new scores (positions already swapped)
        for other_idx in 0..n {
            if other_idx != val_idx1 && other_idx != val_idx2 {
                delta += self.get_pair_score(val_idx1, other_idx);
                delta += self.get_pair_score(val_idx2, other_idx);
            }
        }
        delta += self.get_pair_score(val_idx1, val_idx2);

        // Update total score
        self.total_score += delta;
    }

    /// Undo a swap (used when rejecting a move)
    fn undo_swap(&mut self, val_idx1: usize, val_idx2: usize) {
        // Swap positions back
        self.positions.swap(val_idx1, val_idx2);

        // Restore old score (much faster than recalculating)
        self.total_score = self.old_score;
    }
}

/// Calculate toroidal (wrapped) distance component between two coordinates
#[inline]
pub fn toroidal_distance_component(a: usize, b: usize, size: usize) -> i32 {
    let diff = (a as i32 - b as i32).abs();
    diff.min(size as i32 - diff)
}

#[derive(Clone)]
pub struct Kernel {
    pub grid: Vec<f64>,
    pub size: usize,
}

impl Kernel {
    pub fn new(size: usize, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), size * size, "Values must match grid size");
        Kernel { grid: values, size }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.grid[row * self.size + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.grid[row * self.size + col] = value;
    }

    /// Build position map: value -> flat position index
    /// Values must be integers in range [1, n] where n = grid.len()
    pub fn build_positions(&self) -> Vec<usize> {
        let n = self.grid.len();
        let mut positions = vec![0; n];
        let mut seen = vec![false; n];

        for row in 0..self.size {
            for col in 0..self.size {
                let value = self.get(row, col);

                // Values must be 1-based integers: 1.0, 2.0, ..., n
                assert!(value >= 1.0 && value <= n as f64 && value.fract() == 0.0,
                        "Grid values must be integers in range [1, {}], found {}", n, value);

                let index = (value as usize) - 1;

                assert!(!seen[index], "Duplicate value {} found in grid", value);
                seen[index] = true;

                positions[index] = row * self.size + col;
            }
        }

        positions
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

/// Optimize kernel using simulated annealing with optional auto-calibration
///
/// If initial_temp is 0.0, it will be auto-calibrated based on typical score deltas.
/// If cooling_rate is 0.0, it will be computed to reach 1% of initial temp by the end.
/// sequence_weight_strength: 0.0 = no sequence penalty, 1.0 = full 1/distance penalty (default)
pub fn optimize_kernel(
    size: usize,
    values: Vec<f64>,
    iterations: usize,
    initial_temp: f64,
    cooling_rate: f64,
    seed: u64,
    sequence_weight_strength: f64,
) -> Kernel {
    let mut initial_temp = initial_temp;
    let mut cooling_rate = cooling_rate;
    assert_eq!(values.len(), size * size, "Values must match grid size");

    // Simple pseudo-random number generator (LCG)
    let mut rng_state = seed;
    let mut random = || {
        rng_state = rng_state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);
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
    let mut scorer = IncrementalScorer::new_with_sequence_weight(size, positions, sequence_weight_strength);
    let mut current_score = scorer.total_score();

    let mut best_arrangement = current.clone();
    let mut best_score = current_score;

    // Auto-calibrate temperature if initial_temp is 0.0
    if initial_temp == 0.0 {
        println!("Auto-calibrating initial temperature...");
        let calibration_samples = CALIBRATION_SAMPLES.min(size * size * 10);
        let mut delta_sum = 0.0;
        let mut delta_count = 0;

        for _ in 0..calibration_samples {
            let i = (random() as usize) % current.len();
            let j = (random() as usize) % (current.len() - 1);
            let j = if j >= i { j + 1 } else { j };

            let val_idx1 = current[i] as usize - 1;
            let val_idx2 = current[j] as usize - 1;

            current.swap(i, j);
            scorer.update_after_swap(val_idx1, val_idx2);
            let new_score = scorer.total_score();
            let delta = (new_score - current_score).abs();

            if delta > 0.0 {
                delta_sum += delta;
                delta_count += 1;
            }

            // Undo the swap
            current.swap(i, j);
            scorer.undo_swap(val_idx1, val_idx2);
        }

        if delta_count > 0 {
            let avg_delta = delta_sum / delta_count as f64;
            initial_temp = avg_delta * CALIBRATION_TEMP_MULTIPLIER;
            println!("  Average delta: {:.4}, setting initial_temp = {:.4}", avg_delta, initial_temp);
        } else {
            println!("  No valid deltas found, using default initial_temp = {:.4}", initial_temp);
        }

        // Calculate cooling rate to reach 1% of initial temp by end
        cooling_rate = TARGET_FINAL_TEMP_RATIO.powf(1.0 / iterations as f64);
        println!("  Calculated cooling_rate = {:.6} (will reach {:.0}% by iteration {})", cooling_rate, TARGET_FINAL_TEMP_RATIO * 100.0, iterations);
    } else if cooling_rate == 0.0 {
        // Just compute cooling rate from a provided initial_temp
        cooling_rate = TARGET_FINAL_TEMP_RATIO.powf(1.0 / iterations as f64);
        println!("Computed cooling_rate = {:.6} (will reach {:.0}% by iteration {})", cooling_rate, TARGET_FINAL_TEMP_RATIO * 100.0, iterations);
    }

    let mut temperature = initial_temp;
    let start_time = Instant::now();

    // Tracking for reporting
    let mut accepts = 0;
    let mut rejects = 0;

    println!("Starting simulated annealing...");
    println!("Initial score: {:.2}, Temp: {:.4}, Cooling: {:.6}", current_score, temperature, cooling_rate);

    for iteration in 0..iterations {
        // Progress reporting
        if iteration % REPORT_INTERVAL == 0 || iteration == iterations - 1 {
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
            accepts = 0;
            rejects = 0;
        }

        // Pick two random positions to swap (ensure different)
        let i = (random() as usize) % current.len();
        let j = (random() as usize) % (current.len() - 1);
        let j = if j >= i { j + 1 } else { j };

        // Get value indices for the positions we're swapping
        let val_idx1 = current[i] as usize - 1;
        let val_idx2 = current[j] as usize - 1;

        // Try swap in grid
        current.swap(i, j);

        // Update scorer incrementally
        scorer.update_after_swap(val_idx1, val_idx2);
        let new_score = scorer.total_score();

        let delta = new_score - current_score;

        // Accept if better, or sometimes if worse based on temperature
        let accept = if delta > 0.0 {
            true
        } else {
            // Fast random float in [0, 1): use top 53 bits for mantissa
            let random_bits = random() >> 11;
            let random_val = (random_bits as f64) * (1.0 / 9007199254740992.0); // 2^53
            random_val < (delta / temperature).exp()
        };

        if accept {
            // Accept the swap
            current_score = new_score;
            accepts += 1;

            if current_score > best_score {
                best_score = current_score;
                best_arrangement.copy_from_slice(&current);
            }
        } else {
            // Reject: swap back both current and scorer
            current.swap(i, j);
            scorer.undo_swap(val_idx1, val_idx2);
            rejects += 1;
        }

        // Cool down
        temperature *= cooling_rate;
    }

    println!("\nSimulated annealing complete!");
    println!("Best score found: {:.2}", best_score);
    println!("Total time: {:.2}s", start_time.elapsed().as_secs_f64());

    Kernel::new(size, best_arrangement)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2x2_scoring() {
        let kernel1 = Kernel::new(2, vec![1.0, 2.0, 3.0, 4.0]);
        let kernel2 = Kernel::new(2, vec![1.0, 4.0, 3.0, 2.0]);

        let positions1 = kernel1.build_positions();
        let positions2 = kernel2.build_positions();

        let scorer1 = IncrementalScorer::new(2, positions1);
        let scorer2 = IncrementalScorer::new(2, positions2);

        let score1 = scorer1.total_score();
        let score2 = scorer2.total_score();

        println!("Kernel 1 score: {:.2}", score1);
        println!("Kernel 2 score: {:.2}", score2);

        assert!(score2 > score1, "Kernel 2 should score higher");
    }

    #[test]
    fn test_4x4_optimization() {
        let values: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let kernel = optimize_kernel(4, values, 100_000, 0.0, 0.0, 12345, 1.0);

        println!("Optimized 4x4 arrangement:");
        println!("{}", kernel);

        let positions = kernel.build_positions();
        let scorer = IncrementalScorer::new_with_sequence_weight(4, positions, 1.0);
        println!("Score: {:.2}", scorer.total_score());
    }

    #[test]
    fn test_manual_vs_auto_calibration() {
        let values: Vec<f64> = (1..=16).map(|x| x as f64).collect();

        println!("\n=== Manual parameters ===");
        let kernel1 = optimize_kernel(4, values.clone(), 50_000, 1.0, 0.9999, 12345, 1.0);
        let positions1 = kernel1.build_positions();
        let scorer1 = IncrementalScorer::new(4, positions1);
        let score1 = scorer1.total_score();

        println!("\n=== Auto-calibrated ===");
        let kernel2 = optimize_kernel(4, values, 50_000, 0.0, 0.0, 12345, 1.0);
        let positions2 = kernel2.build_positions();
        let scorer2 = IncrementalScorer::new(4, positions2);
        let score2 = scorer2.total_score();

        println!("\nScore with manual params: {:.2}", score1);
        println!("Score with auto-calibration: {:.2}", score2);
    }

    #[test]
    fn test_sequence_weight_impact() {
        let values: Vec<f64> = (1..=16).map(|x| x as f64).collect();

        println!("\n=== No sequence weight (strength=0.0) ===");
        let kernel1 = optimize_kernel(4, values.clone(), 50_000, 0.0, 0.0, 12345, 0.0);
        let positions1 = kernel1.build_positions();
        let scorer1 = IncrementalScorer::new_with_sequence_weight(4, positions1, 0.0);
        let score1 = scorer1.total_score();

        println!("\n=== Half sequence weight (strength=0.5) ===");
        let kernel2 = optimize_kernel(4, values.clone(), 50_000, 0.0, 0.0, 12345, 0.5);
        let positions2 = kernel2.build_positions();
        let scorer2 = IncrementalScorer::new_with_sequence_weight(4, positions2, 0.5);
        let score2 = scorer2.total_score();

        println!("\n=== Full sequence weight (strength=1.0) ===");
        let kernel3 = optimize_kernel(4, values, 50_000, 0.0, 0.0, 12345, 1.0);
        let positions3 = kernel3.build_positions();
        let scorer3 = IncrementalScorer::new_with_sequence_weight(4, positions3, 1.0);
        let score3 = scorer3.total_score();

        println!("\nScore with no sequence weight: {:.2}", score1);
        println!("Score with half sequence weight: {:.2}", score2);
        println!("Score with full sequence weight: {:.2}", score3);
    }
}
