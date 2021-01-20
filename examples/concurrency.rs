//! Show how to use a FFT in multiple threads

use std::sync::Arc;
use std::thread;

use realfft::RealFftPlanner;

fn main() {
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(100);

    let threads: Vec<thread::JoinHandle<_>> = (0..2)
        .map(|_| {
            let fft_copy = Arc::clone(&fft);
            thread::spawn(move || {
                let mut data = fft_copy.make_input_vec();
                let mut output = fft_copy.make_output_vec();
                fft_copy.process(&mut data, &mut output).unwrap();
            })
        })
        .collect();

    for thread in threads {
        thread.join().unwrap();
    }
}
