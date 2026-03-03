use image_filters::filters::ThresholdKernel;

fn main() {
    println!("Testing kernel cache with seed strings...\n");

    let size = 4;

    // Test 1: Alphanumeric seed
    println!("=== Test 1: Alphanumeric seed 'mykernel' ===");
    let seed_str = "mykernel";
    let seed = ThresholdKernel::seed_from_string(seed_str);
    println!("String '{}' hashes to numeric seed: {}", seed_str, seed);

    println!("\nFirst call with 'mykernel':");
    let start = std::time::Instant::now();
    let _kernel1 = ThresholdKernel::from_annealing_with_string(size, size, seed, Some(seed_str.to_string()));
    println!("Time: {:?}", start.elapsed());

    println!("\nSecond call with 'mykernel' (should load from cache):");
    let start = std::time::Instant::now();
    let _kernel2 = ThresholdKernel::from_annealing_with_string(size, size, seed, Some(seed_str.to_string()));
    println!("Time: {:?}", start.elapsed());

    // Test 2: Numeric seed as string
    println!("\n=== Test 2: Numeric seed '12345' ===");
    let seed_str2 = "12345";
    let seed2: u64 = seed_str2.parse().unwrap();

    println!("\nFirst call with '12345':");
    let start = std::time::Instant::now();
    let _kernel3 = ThresholdKernel::from_annealing_with_string(size, size, seed2, Some(seed_str2.to_string()));
    println!("Time: {:?}", start.elapsed());

    println!("\nSecond call with '12345' (should load from cache):");
    let start = std::time::Instant::now();
    let _kernel4 = ThresholdKernel::from_annealing_with_string(size, size, seed2, Some(seed_str2.to_string()));
    println!("Time: {:?}", start.elapsed());

    // Test 3: Verify we can look up by numeric seed too
    println!("\n=== Test 3: Looking up '12345' by numeric seed only ===");
    let start = std::time::Instant::now();
    let _kernel5 = ThresholdKernel::from_annealing_with_string(size, size, seed2, None);
    println!("Time: {:?}", start.elapsed());

    println!("\n=== All tests complete! ===");
    println!("\nCheck kernels/ directory:");
    if let Ok(entries) = std::fs::read_dir("kernels") {
        for entry in entries.flatten() {
            println!("  - {}", entry.file_name().to_string_lossy());
        }
    }
}
