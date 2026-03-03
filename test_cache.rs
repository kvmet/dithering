use image_filters::filters::ThresholdKernel;

fn main() {
    println!("Testing kernel cache...\n");

    let size = 4;
    let seed = 12345;

    println!("First call - should compute and cache:");
    let start = std::time::Instant::now();
    let _kernel1 = ThresholdKernel::from_annealing(size, size, seed);
    let duration1 = start.elapsed();
    println!("Time: {:?}\n", duration1);

    println!("Second call - should load from cache:");
    let start = std::time::Instant::now();
    let _kernel2 = ThresholdKernel::from_annealing(size, size, seed);
    let duration2 = start.elapsed();
    println!("Time: {:?}\n", duration2);

    println!("Speedup: {:.1}x faster", duration1.as_secs_f64() / duration2.as_secs_f64());

    println!("\nTrying a different seed:");
    let seed2 = 99999;
    let start = std::time::Instant::now();
    let _kernel3 = ThresholdKernel::from_annealing(size, size, seed2);
    let duration3 = start.elapsed();
    println!("Time: {:?}", duration3);
}
