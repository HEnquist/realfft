//! Show that compiler does not allow incorrect output fft buffers

use realfft::{num_complex::Complex, RealFftPlanner};

fn main() {
    let mut planner = RealFftPlanner::<f32>::new();
    const LENGTH: usize = 100;
    let r2c = planner.plan_fft_forward::<LENGTH>();
    let mut input = [0.0; LENGTH];
    let mut output = [Complex::default(); LENGTH / 2];
    r2c.process(&mut input, &mut output).unwrap();
}
