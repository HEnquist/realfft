//! Show that compiler does not allow incorrect input fft buffers

use realfft::{num_complex::Complex, RealFftPlanner};

fn main() {
    let mut planner = RealFftPlanner::<f32>::new();
    const LENGTH: usize = 100;
    let r2c = planner.plan_fft_forward::<LENGTH>();
    let mut input = [0.0; LENGTH - 1];
    let mut output = [Complex::default(); LENGTH / 2 + 1];
    r2c.process(&mut input, &mut output).unwrap();
}
