use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
extern crate realfft;
extern crate rustfft;

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_fft(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(len);
    let mut scratch = vec![Complex::from(0.0); fft.get_outofplace_scratch_len()];

    let mut signal = vec![
        Complex {
            re: 0_f64,
            im: 0_f64
        };
        len
    ];
    let mut spectrum = signal.clone();
    b.iter(|| fft.process_outofplace_with_scratch(&mut signal, &mut spectrum, &mut scratch));
}

fn bench_realfft(b: &mut Bencher, len: usize) {
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(len);

    let mut signal = vec![0_f64; len];
    let mut spectrum = vec![
        Complex {
            re: 0_f64,
            im: 0_f64
        };
        len / 2 + 1
    ];
    let mut scratch = vec![Complex::from(0.0); fft.get_scratch_len()];
    b.iter(|| fft.process_with_scratch(&mut signal, &mut spectrum, &mut scratch));
}

/// Times just the FFT execution (not allocation and pre-calculation)
/// for a given length
fn bench_ifft(b: &mut Bencher, len: usize) {
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_inverse(len);
    let mut scratch = vec![Complex::from(0.0); fft.get_outofplace_scratch_len()];

    let mut signal = vec![
        Complex {
            re: 0_f64,
            im: 0_f64
        };
        len
    ];
    let mut spectrum = signal.clone();
    b.iter(|| fft.process_outofplace_with_scratch(&mut signal, &mut spectrum, &mut scratch));
}

fn bench_realifft(b: &mut Bencher, len: usize) {
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(len);

    let mut signal = vec![0_f64; len];
    let mut spectrum = vec![
        Complex {
            re: 0_f64,
            im: 0_f64
        };
        len / 2 + 1
    ];
    let mut scratch = vec![Complex::from(0.0); fft.get_scratch_len()];
    b.iter(|| fft.process_with_scratch(&mut spectrum, &mut signal, &mut scratch));
}

fn bench_pow2_fw(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fw Powers of 2");
    for i in [8, 16, 32, 64, 128, 256, 1024, 4096, 65536].iter() {
        group.bench_with_input(BenchmarkId::new("Complex", i), i, |b, i| bench_fft(b, *i));
        group.bench_with_input(BenchmarkId::new("Real", i), i, |b, i| bench_realfft(b, *i));
    }
    group.finish();
}

fn bench_pow2_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inv Powers of 2");
    for i in [8, 16, 32, 64, 128, 256, 1024, 4096, 65536].iter() {
        group.bench_with_input(BenchmarkId::new("Complex", i), i, |b, i| bench_ifft(b, *i));
        group.bench_with_input(BenchmarkId::new("Real", i), i, |b, i| bench_realifft(b, *i));
    }
    group.finish();
}

//fn bench_pow7(c: &mut Criterion) {
//    let mut group = c.benchmark_group("Powers of 7");
//    for i in [2 * 343, 2 * 2401, 2 * 16807].iter() {
//        group.bench_with_input(BenchmarkId::new("Complex", i), i, |b, i| bench_fft(b, *i));
//        group.bench_with_input(BenchmarkId::new("Real", i), i, |b, i| bench_realfft(b, *i));
//    }
//    group.finish();
//}

fn bench_range_fw(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fw Range 1022-1025");
    for i in 1022..1026 {
        group.bench_with_input(BenchmarkId::new("Complex", i), &i, |b, i| bench_fft(b, *i));
        group.bench_with_input(BenchmarkId::new("Real", i), &i, |b, i| bench_realfft(b, *i));
    }
    group.finish();
}

fn bench_range_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inv Range 1022-1025");
    for i in 1022..1026 {
        group.bench_with_input(BenchmarkId::new("Complex", i), &i, |b, i| bench_ifft(b, *i));
        group.bench_with_input(BenchmarkId::new("Real", i), &i, |b, i| {
            bench_realifft(b, *i)
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pow2_fw,
    bench_range_fw,
    bench_pow2_inv,
    bench_range_inv
);

criterion_main!(benches);
