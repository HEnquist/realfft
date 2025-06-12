#![doc = include_str!("../README.md")]

pub use rustfft::num_complex;
pub use rustfft::num_traits;
pub use rustfft::FftNum;

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;
use std::collections::HashMap;
use std::error;
use std::fmt;
use std::sync::Arc;

type Res<T> = Result<T, FftError>;

/// Custom error returned by FFTs
pub enum FftError {
    /// The input buffer has the wrong size. The transform was not performed.
    ///
    /// The first member of the tuple is the expected size and the second member is the received
    /// size.
    InputBuffer(usize, usize),
    /// The output buffer has the wrong size. The transform was not performed.
    ///
    /// The first member of the tuple is the expected size and the second member is the received
    /// size.
    OutputBuffer(usize, usize),
    /// The scratch buffer has the wrong size. The transform was not performed.
    ///
    /// The first member of the tuple is the minimum size and the second member is the received
    /// size.
    ScratchBuffer(usize, usize),
    /// The input data contained a non-zero imaginary part where there should have been a zero.
    /// The transform was performed, but the result may not be correct.
    ///
    /// The first member of the tuple represents the first index of the complex buffer and the
    /// second member represents the last index of the complex buffer. The values are set to true
    /// if the corresponding complex value contains a non-zero imaginary part.
    InputValues(bool, bool),
}

impl FftError {
    fn fmt_internal(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let desc = match self {
            Self::InputBuffer(expected, got) => {
                format!("Wrong length of input, expected {}, got {}", expected, got)
            }
            Self::OutputBuffer(expected, got) => {
                format!("Wrong length of output, expected {}, got {}", expected, got)
            }
            Self::ScratchBuffer(expected, got) => {
                format!(
                    "Scratch buffer of size {} is too small, must be at least {} long",
                    got, expected
                )
            }
            Self::InputValues(first, last) => match (first, last) {
                (true, false) => "Imaginary part of first value was non-zero.".to_string(),
                (false, true) => "Imaginary part of last value was non-zero.".to_string(),
                (true, true) => {
                    "Imaginary parts of both first and last values were non-zero.".to_string()
                }
                (false, false) => unreachable!(),
            },
        };
        write!(f, "{}", desc)
    }
}

impl fmt::Debug for FftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_internal(f)
    }
}

impl fmt::Display for FftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_internal(f)
    }
}

impl error::Error for FftError {}

fn compute_twiddle<T: FftNum>(index: usize, fft_len: usize) -> Complex<T> {
    let constant = -2f64 * std::f64::consts::PI / fft_len as f64;
    let angle = constant * index as f64;
    Complex {
        re: T::from_f64(angle.cos()).unwrap(),
        im: T::from_f64(angle.sin()).unwrap(),
    }
}

pub struct RealToComplexOdd<T> {
    length: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<T>>,
    scratch_len: usize,
}

pub struct RealToComplexEven<T> {
    twiddles: Vec<Complex<T>>,
    length: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<T>>,
    scratch_len: usize,
}

pub struct ComplexToRealOdd<T> {
    length: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<T>>,
    scratch_len: usize,
}

pub struct ComplexToRealEven<T> {
    twiddles: Vec<Complex<T>>,
    length: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<T>>,
    scratch_len: usize,
}

/// A forward FFT that takes a real-valued input signal of length N
/// and transforms it to a complex spectrum of length N/2+1.
#[allow(clippy::len_without_is_empty)]
pub trait RealToComplex<T>: Sync + Send {
    /// Transform a signal of N real-valued samples,
    /// storing the resulting complex spectrum in the N/2+1
    /// (with N/2 rounded down) element long output slice.
    /// The input buffer is used as scratch space,
    /// so the contents of input should be considered garbage after calling.
    /// It also allocates additional scratch space as needed.
    /// An error is returned if any of the given slices has the wrong length.
    fn process(&self, input: &mut [T], output: &mut [Complex<T>]) -> Res<()>;

    /// Transform a signal of N real-valued samples,
    /// similar to [`process()`](RealToComplex::process).
    /// The difference is that this method uses the provided
    /// scratch buffer instead of allocating new scratch space.
    /// This is faster if the same scratch buffer is used for multiple calls.
    fn process_with_scratch(
        &self,
        input: &mut [T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Res<()>;

    /// Get the minimum length of the scratch buffer needed for `process_with_scratch`.
    fn get_scratch_len(&self) -> usize;

    /// The FFT length.
    /// Get the length of the real signal that this FFT takes as input.
    fn len(&self) -> usize;

    /// Get the number of complex data points that this FFT returns.
    fn complex_len(&self) -> usize {
        self.len() / 2 + 1
    }

    /// Convenience method to make an input vector of the right type and length.
    fn make_input_vec(&self) -> Vec<T>;

    /// Convenience method to make an output vector of the right type and length.
    fn make_output_vec(&self) -> Vec<Complex<T>>;

    /// Convenience method to make a scratch vector of the right type and length.
    fn make_scratch_vec(&self) -> Vec<Complex<T>>;
}

/// An inverse FFT that takes a complex spectrum of length N/2+1
/// and transforms it to a real-valued signal of length N.
#[allow(clippy::len_without_is_empty)]
pub trait ComplexToReal<T>: Sync + Send {
    /// Inverse transform a complex spectrum corresponding to a real-valued signal of length N.
    /// The input is a slice of complex values with length N/2+1 (with N/2 rounded down).
    /// The resulting real-valued signal is stored in the output slice of length N.
    /// The input buffer is used as scratch space,
    /// so the contents of input should be considered garbage after calling.
    /// It also allocates additional scratch space as needed.
    /// An error is returned if any of the given slices has the wrong length.
    /// If the input data is invalid, meaning that one of the positions that should
    /// contain a zero holds a non-zero value, the transform is still performed.
    /// The function then returns an `FftError::InputValues` error to tell that the
    /// result may not be correct.
    fn process(&self, input: &mut [Complex<T>], output: &mut [T]) -> Res<()>;

    /// Inverse transform a complex spectrum,
    /// similar to [`process()`](ComplexToReal::process).
    /// The difference is that this method uses the provided
    /// scratch buffer instead of allocating new scratch space.
    /// This is faster if the same scratch buffer is used for multiple calls.
    fn process_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Res<()>;

    /// Get the minimum length of the scratch space needed for `process_with_scratch`.
    fn get_scratch_len(&self) -> usize;

    /// The FFT length.
    /// Get the length of the real-valued signal that this FFT returns.
    fn len(&self) -> usize;

    /// Get the length of the slice slice of complex values that this FFT accepts as input.
    fn complex_len(&self) -> usize {
        self.len() / 2 + 1
    }

    /// Convenience method to make an input vector of the right type and length.
    fn make_input_vec(&self) -> Vec<Complex<T>>;

    /// Convenience method to make an output vector of the right type and length.
    fn make_output_vec(&self) -> Vec<T>;

    /// Convenience method to make a scratch vector of the right type and length.
    fn make_scratch_vec(&self) -> Vec<Complex<T>>;
}

fn zip3<A, B, C>(a: A, b: B, c: C) -> impl Iterator<Item = (A::Item, B::Item, C::Item)>
where
    A: IntoIterator,
    B: IntoIterator,
    C: IntoIterator,
{
    a.into_iter()
        .zip(b.into_iter().zip(c))
        .map(|(x, (y, z))| (x, y, z))
}

/// A planner is used to create FFTs.
/// It caches results internally,
/// so when making more than one FFT it is advisable to reuse the same planner.
pub struct RealFftPlanner<T: FftNum> {
    planner: FftPlanner<T>,
    r2c_cache: HashMap<usize, Arc<dyn RealToComplex<T>>>,
    c2r_cache: HashMap<usize, Arc<dyn ComplexToReal<T>>>,
}

impl<T: FftNum> RealFftPlanner<T> {
    /// Create a new planner.
    pub fn new() -> Self {
        let planner = FftPlanner::<T>::new();
        Self {
            r2c_cache: HashMap::new(),
            c2r_cache: HashMap::new(),
            planner,
        }
    }

    /// Plan a real-to-complex forward FFT. Returns the FFT in a shared reference.
    /// If requesting a second forward FFT of the same length,
    /// the planner will return a new reference to the already existing one.
    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn RealToComplex<T>> {
        if let Some(fft) = self.r2c_cache.get(&len) {
            Arc::clone(fft)
        } else {
            let fft = if len % 2 > 0 {
                Arc::new(RealToComplexOdd::new(len, &mut self.planner)) as Arc<dyn RealToComplex<T>>
            } else {
                Arc::new(RealToComplexEven::new(len, &mut self.planner))
                    as Arc<dyn RealToComplex<T>>
            };
            self.r2c_cache.insert(len, Arc::clone(&fft));
            fft
        }
    }

    /// Plan a complex-to-real inverse FFT. Returns the FFT in a shared reference.
    /// If requesting a second inverse FFT of the same length,
    /// the planner will return a new reference to the already existing one.
    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn ComplexToReal<T>> {
        if let Some(fft) = self.c2r_cache.get(&len) {
            Arc::clone(fft)
        } else {
            let fft = if len % 2 > 0 {
                Arc::new(ComplexToRealOdd::new(len, &mut self.planner)) as Arc<dyn ComplexToReal<T>>
            } else {
                Arc::new(ComplexToRealEven::new(len, &mut self.planner))
                    as Arc<dyn ComplexToReal<T>>
            };
            self.c2r_cache.insert(len, Arc::clone(&fft));
            fft
        }
    }
}

impl<T: FftNum> Default for RealFftPlanner<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FftNum> RealToComplexOdd<T> {
    /// Create a new RealToComplex forward FFT for real-valued input data of a given length,
    /// and uses the given FftPlanner to build the inner FFT.
    /// Panics if the length is not odd.
    pub fn new(length: usize, fft_planner: &mut FftPlanner<T>) -> Self {
        if length % 2 == 0 {
            panic!("Length must be odd, got {}", length,);
        }
        let fft = fft_planner.plan_fft_forward(length);
        let scratch_len = fft.get_inplace_scratch_len() + length;
        RealToComplexOdd {
            length,
            fft,
            scratch_len,
        }
    }
}

impl<T: FftNum> RealToComplex<T> for RealToComplexOdd<T> {
    fn process(&self, input: &mut [T], output: &mut [Complex<T>]) -> Res<()> {
        let mut scratch = self.make_scratch_vec();
        self.process_with_scratch(input, output, &mut scratch)
    }

    fn process_with_scratch(
        &self,
        input: &mut [T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Res<()> {
        if input.len() != self.length {
            return Err(FftError::InputBuffer(self.length, input.len()));
        }
        let expected_output_buffer_size = self.complex_len();
        if output.len() != expected_output_buffer_size {
            return Err(FftError::OutputBuffer(
                expected_output_buffer_size,
                output.len(),
            ));
        }
        if scratch.len() < (self.scratch_len) {
            return Err(FftError::ScratchBuffer(self.scratch_len, scratch.len()));
        }
        let (buffer, fft_scratch) = scratch.split_at_mut(self.length);

        for (val, buf) in input.iter().zip(buffer.iter_mut()) {
            *buf = Complex::new(*val, T::zero());
        }
        // FFT and store result in buffer_out
        self.fft.process_with_scratch(buffer, fft_scratch);
        output.copy_from_slice(&buffer[0..self.complex_len()]);
        if let Some(elem) = output.first_mut() {
            elem.im = T::zero();
        }
        Ok(())
    }

    fn get_scratch_len(&self) -> usize {
        self.scratch_len
    }

    fn len(&self) -> usize {
        self.length
    }

    fn make_input_vec(&self) -> Vec<T> {
        vec![T::zero(); self.len()]
    }

    fn make_output_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::zero(); self.complex_len()]
    }

    fn make_scratch_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::zero(); self.get_scratch_len()]
    }
}

impl<T: FftNum> RealToComplexEven<T> {
    /// Create a new RealToComplex forward FFT for real-valued input data of a given length,
    /// and uses the given FftPlanner to build the inner FFT.
    /// Panics if the length is not even.
    pub fn new(length: usize, fft_planner: &mut FftPlanner<T>) -> Self {
        if length % 2 > 0 {
            panic!("Length must be even, got {}", length,);
        }
        let twiddle_count = if length % 4 == 0 {
            length / 4
        } else {
            length / 4 + 1
        };
        let twiddles: Vec<Complex<T>> = (1..twiddle_count)
            .map(|i| compute_twiddle(i, length) * T::from_f64(0.5).unwrap())
            .collect();
        let fft = fft_planner.plan_fft_forward(length / 2);
        let scratch_len = fft.get_outofplace_scratch_len();
        RealToComplexEven {
            twiddles,
            length,
            fft,
            scratch_len,
        }
    }
}

impl<T: FftNum> RealToComplex<T> for RealToComplexEven<T> {
    fn process(&self, input: &mut [T], output: &mut [Complex<T>]) -> Res<()> {
        let mut scratch = self.make_scratch_vec();
        self.process_with_scratch(input, output, &mut scratch)
    }

    fn process_with_scratch(
        &self,
        input: &mut [T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Res<()> {
        if input.len() != self.length {
            return Err(FftError::InputBuffer(self.length, input.len()));
        }
        let expected_output_buffer_size = self.complex_len();
        if output.len() != expected_output_buffer_size {
            return Err(FftError::OutputBuffer(
                expected_output_buffer_size,
                output.len(),
            ));
        }
        if scratch.len() < (self.scratch_len) {
            return Err(FftError::ScratchBuffer(self.scratch_len, scratch.len()));
        }

        let fftlen = self.length / 2;
        let buf_in = unsafe {
            let ptr = input.as_mut_ptr() as *mut Complex<T>;
            let len = input.len();
            std::slice::from_raw_parts_mut(ptr, len / 2)
        };

        // FFT and store result in buffer_out
        self.fft
            .process_outofplace_with_scratch(buf_in, &mut output[0..fftlen], scratch);
        let (mut output_left, mut output_right) = output.split_at_mut(output.len() / 2);

        // The first and last element don't require any twiddle factors, so skip that work
        match (output_left.first_mut(), output_right.last_mut()) {
            (Some(first_element), Some(last_element)) => {
                // The first and last elements are just a sum and difference of the first value's real and imaginary values
                let first_value = *first_element;
                *first_element = Complex {
                    re: first_value.re + first_value.im,
                    im: T::zero(),
                };
                *last_element = Complex {
                    re: first_value.re - first_value.im,
                    im: T::zero(),
                };

                // Chop the first and last element off of our slices so that the loop below doesn't have to deal with them
                output_left = &mut output_left[1..];
                let right_len = output_right.len();
                output_right = &mut output_right[..right_len - 1];
            }
            _ => {
                return Ok(());
            }
        }
        // Loop over the remaining elements and apply twiddle factors on them
        for (twiddle, out, out_rev) in zip3(
            self.twiddles.iter(),
            output_left.iter_mut(),
            output_right.iter_mut().rev(),
        ) {
            let sum = *out + *out_rev;
            let diff = *out - *out_rev;
            let half = T::from_f64(0.5).unwrap();
            // Apply twiddle factors. Theoretically we'd have to load 2 separate twiddle factors here, one for the beginning
            // and one for the end. But the twiddle factor for the end is just the twiddle for the beginning, with the
            // real part negated. Since it's the same twiddle, we can factor out a ton of math ops and cut the number of
            // multiplications in half.
            let twiddled_re_sum = sum * twiddle.re;
            let twiddled_im_sum = sum * twiddle.im;
            let twiddled_re_diff = diff * twiddle.re;
            let twiddled_im_diff = diff * twiddle.im;
            let half_sum_re = half * sum.re;
            let half_diff_im = half * diff.im;

            let output_twiddled_real = twiddled_re_sum.im + twiddled_im_diff.re;
            let output_twiddled_im = twiddled_im_sum.im - twiddled_re_diff.re;

            // We finally have all the data we need to write the transformed data back out where we found it.
            *out = Complex {
                re: half_sum_re + output_twiddled_real,
                im: half_diff_im + output_twiddled_im,
            };

            *out_rev = Complex {
                re: half_sum_re - output_twiddled_real,
                im: output_twiddled_im - half_diff_im,
            };
        }

        // If the output len is odd, the loop above can't postprocess the centermost element, so handle that separately.
        if output.len() % 2 == 1 {
            if let Some(center_element) = output.get_mut(output.len() / 2) {
                center_element.im = -center_element.im;
            }
        }
        Ok(())
    }
    fn get_scratch_len(&self) -> usize {
        self.scratch_len
    }

    fn len(&self) -> usize {
        self.length
    }

    fn make_input_vec(&self) -> Vec<T> {
        vec![T::zero(); self.len()]
    }

    fn make_output_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::zero(); self.complex_len()]
    }

    fn make_scratch_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::zero(); self.get_scratch_len()]
    }
}

impl<T: FftNum> ComplexToRealOdd<T> {
    /// Create a new ComplexToRealOdd inverse FFT for complex input spectra.
    /// The `length` parameter refers to the length of the resulting real-valued signal.
    /// Uses the given FftPlanner to build the inner FFT.
    /// Panics if the length is not odd.
    pub fn new(length: usize, fft_planner: &mut FftPlanner<T>) -> Self {
        if length % 2 == 0 {
            panic!("Length must be odd, got {}", length,);
        }
        let fft = fft_planner.plan_fft_inverse(length);
        let scratch_len = length + fft.get_inplace_scratch_len();
        ComplexToRealOdd {
            length,
            fft,
            scratch_len,
        }
    }
}

impl<T: FftNum> ComplexToReal<T> for ComplexToRealOdd<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [T]) -> Res<()> {
        let mut scratch = self.make_scratch_vec();
        self.process_with_scratch(input, output, &mut scratch)
    }

    fn process_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Res<()> {
        let expected_input_buffer_size = self.complex_len();
        if input.len() != expected_input_buffer_size {
            return Err(FftError::InputBuffer(
                expected_input_buffer_size,
                input.len(),
            ));
        }
        if output.len() != self.length {
            return Err(FftError::OutputBuffer(self.length, output.len()));
        }
        if scratch.len() < (self.scratch_len) {
            return Err(FftError::ScratchBuffer(self.scratch_len, scratch.len()));
        }

        let first_invalid = if input[0].im != T::from_f64(0.0).unwrap() {
            input[0].im = T::from_f64(0.0).unwrap();
            true
        } else {
            false
        };

        let (buffer, fft_scratch) = scratch.split_at_mut(self.length);

        buffer[0..input.len()].copy_from_slice(input);
        for (buf, val) in buffer
            .iter_mut()
            .rev()
            .take(self.length / 2)
            .zip(input.iter().skip(1))
        {
            *buf = val.conj();
        }
        self.fft.process_with_scratch(buffer, fft_scratch);
        for (val, out) in buffer.iter().zip(output.iter_mut()) {
            *out = val.re;
        }
        if first_invalid {
            return Err(FftError::InputValues(true, false));
        }
        Ok(())
    }

    fn get_scratch_len(&self) -> usize {
        self.scratch_len
    }

    fn len(&self) -> usize {
        self.length
    }

    fn make_input_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::zero(); self.complex_len()]
    }

    fn make_output_vec(&self) -> Vec<T> {
        vec![T::zero(); self.len()]
    }

    fn make_scratch_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::zero(); self.get_scratch_len()]
    }
}

impl<T: FftNum> ComplexToRealEven<T> {
    /// Create a new ComplexToRealEven inverse FFT for complex input spectra.
    /// The `length` parameter refers to the length of the resulting real-valued signal.
    /// Uses the given FftPlanner to build the inner FFT.
    /// Panics if the length is not even.
    pub fn new(length: usize, fft_planner: &mut FftPlanner<T>) -> Self {
        if length % 2 > 0 {
            panic!("Length must be even, got {}", length,);
        }
        let twiddle_count = if length % 4 == 0 {
            length / 4
        } else {
            length / 4 + 1
        };
        let twiddles: Vec<Complex<T>> = (1..twiddle_count)
            .map(|i| compute_twiddle(i, length).conj())
            .collect();
        let fft = fft_planner.plan_fft_inverse(length / 2);
        let scratch_len = fft.get_outofplace_scratch_len();
        ComplexToRealEven {
            twiddles,
            length,
            fft,
            scratch_len,
        }
    }
}
impl<T: FftNum> ComplexToReal<T> for ComplexToRealEven<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [T]) -> Res<()> {
        let mut scratch = self.make_scratch_vec();
        self.process_with_scratch(input, output, &mut scratch)
    }

    fn process_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Res<()> {
        let expected_input_buffer_size = self.complex_len();
        if input.len() != expected_input_buffer_size {
            return Err(FftError::InputBuffer(
                expected_input_buffer_size,
                input.len(),
            ));
        }
        if output.len() != self.length {
            return Err(FftError::OutputBuffer(self.length, output.len()));
        }
        if scratch.len() < (self.scratch_len) {
            return Err(FftError::ScratchBuffer(self.scratch_len, scratch.len()));
        }
        if input.is_empty() {
            return Ok(());
        }
        let first_invalid = if input[0].im != T::from_f64(0.0).unwrap() {
            input[0].im = T::from_f64(0.0).unwrap();
            true
        } else {
            false
        };
        let last_invalid = if input[input.len() - 1].im != T::from_f64(0.0).unwrap() {
            input[input.len() - 1].im = T::from_f64(0.0).unwrap();
            true
        } else {
            false
        };

        let (mut input_left, mut input_right) = input.split_at_mut(input.len() / 2);

        // We have to preprocess the input in-place before we send it to the FFT.
        // The first and centermost values have to be preprocessed separately from the rest, so do that now.
        match (input_left.first_mut(), input_right.last_mut()) {
            (Some(first_input), Some(last_input)) => {
                let first_sum = *first_input + *last_input;
                let first_diff = *first_input - *last_input;

                *first_input = Complex {
                    re: first_sum.re - first_sum.im,
                    im: first_diff.re - first_diff.im,
                };

                input_left = &mut input_left[1..];
                let right_len = input_right.len();
                input_right = &mut input_right[..right_len - 1];
            }
            _ => return Ok(()),
        };

        // now, in a loop, preprocess the rest of the elements 2 at a time.
        for (twiddle, fft_input, fft_input_rev) in zip3(
            self.twiddles.iter(),
            input_left.iter_mut(),
            input_right.iter_mut().rev(),
        ) {
            let sum = *fft_input + *fft_input_rev;
            let diff = *fft_input - *fft_input_rev;

            // Apply twiddle factors. Theoretically we'd have to load 2 separate twiddle factors here, one for the beginning
            // and one for the end. But the twiddle factor for the end is just the twiddle for the beginning, with the
            // real part negated. Since it's the same twiddle, we can factor out a ton of math ops and cut the number of
            // multiplications in half.
            let twiddled_re_sum = sum * twiddle.re;
            let twiddled_im_sum = sum * twiddle.im;
            let twiddled_re_diff = diff * twiddle.re;
            let twiddled_im_diff = diff * twiddle.im;

            let output_twiddled_real = twiddled_re_sum.im + twiddled_im_diff.re;
            let output_twiddled_im = twiddled_im_sum.im - twiddled_re_diff.re;

            // We finally have all the data we need to write our preprocessed data back where we got it from.
            *fft_input = Complex {
                re: sum.re - output_twiddled_real,
                im: diff.im - output_twiddled_im,
            };
            *fft_input_rev = Complex {
                re: sum.re + output_twiddled_real,
                im: -output_twiddled_im - diff.im,
            }
        }

        // If the output len is odd, the loop above can't preprocess the centermost element, so handle that separately
        if input.len() % 2 == 1 {
            let center_element = input[input.len() / 2];
            let doubled = center_element + center_element;
            input[input.len() / 2] = doubled.conj();
        }

        // FFT and store result in buffer_out
        let buf_out = unsafe {
            let ptr = output.as_mut_ptr() as *mut Complex<T>;
            let len = output.len();
            std::slice::from_raw_parts_mut(ptr, len / 2)
        };
        self.fft
            .process_outofplace_with_scratch(&mut input[..buf_out.len()], buf_out, scratch);
        if first_invalid || last_invalid {
            return Err(FftError::InputValues(first_invalid, last_invalid));
        }
        Ok(())
    }

    fn get_scratch_len(&self) -> usize {
        self.scratch_len
    }

    fn len(&self) -> usize {
        self.length
    }

    fn make_input_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::zero(); self.complex_len()]
    }

    fn make_output_vec(&self) -> Vec<T> {
        vec![T::zero(); self.len()]
    }

    fn make_scratch_vec(&self) -> Vec<Complex<T>> {
        vec![Complex::zero(); self.get_scratch_len()]
    }
}

#[cfg(test)]
mod tests {
    use crate::FftError;
    use crate::RealFftPlanner;
    use rand::Rng;
    use rustfft::num_complex::Complex;
    use rustfft::num_traits::{Float, Zero};
    use rustfft::FftPlanner;
    use std::error::Error;
    use std::ops::Sub;

    // get the largest difference
    fn compare_complex<T: Float + Sub>(a: &[Complex<T>], b: &[Complex<T>]) -> T {
        a.iter()
            .zip(b.iter())
            .fold(T::zero(), |maxdiff, (val_a, val_b)| {
                let diff = (val_a - val_b).norm();
                if maxdiff > diff {
                    maxdiff
                } else {
                    diff
                }
            })
    }

    // get the largest difference
    fn compare_scalars<T: Float + Sub>(a: &[T], b: &[T]) -> T {
        a.iter()
            .zip(b.iter())
            .fold(T::zero(), |maxdiff, (val_a, val_b)| {
                let diff = (*val_a - *val_b).abs();
                if maxdiff > diff {
                    maxdiff
                } else {
                    diff
                }
            })
    }

    // Compare ComplexToReal with standard inverse FFT
    #[test]
    fn complex_to_real_64() {
        for length in 1..1000 {
            let mut real_planner = RealFftPlanner::<f64>::new();
            let c2r = real_planner.plan_fft_inverse(length);
            let mut out_a = c2r.make_output_vec();
            let mut indata = c2r.make_input_vec();
            let mut rustfft_check: Vec<Complex<f64>> = vec![Complex::zero(); length];
            let mut rng = rand::rng();
            for val in indata.iter_mut() {
                *val = Complex::new(rng.random::<f64>(), rng.random::<f64>());
            }
            indata[0].im = 0.0;
            if length % 2 == 0 {
                indata[length / 2].im = 0.0;
            }
            for (val_long, val) in rustfft_check
                .iter_mut()
                .take(c2r.complex_len())
                .zip(indata.iter())
            {
                *val_long = *val;
            }
            for (val_long, val) in rustfft_check
                .iter_mut()
                .rev()
                .take(length / 2)
                .zip(indata.iter().skip(1))
            {
                *val_long = val.conj();
            }
            let mut fft_planner = FftPlanner::<f64>::new();
            let fft = fft_planner.plan_fft_inverse(length);

            c2r.process(&mut indata, &mut out_a).unwrap();
            fft.process(&mut rustfft_check);

            let check_real = rustfft_check.iter().map(|val| val.re).collect::<Vec<f64>>();
            let maxdiff = compare_scalars(&out_a, &check_real);
            assert!(
                maxdiff < 1.0e-9,
                "Length: {}, too large error: {}",
                length,
                maxdiff
            );
        }
    }

    // Compare ComplexToReal with standard inverse FFT
    #[test]
    fn complex_to_real_32() {
        for length in 1..1000 {
            let mut real_planner = RealFftPlanner::<f32>::new();
            let c2r = real_planner.plan_fft_inverse(length);
            let mut out_a = c2r.make_output_vec();
            let mut indata = c2r.make_input_vec();
            let mut rustfft_check: Vec<Complex<f32>> = vec![Complex::zero(); length];
            let mut rng = rand::rng();
            for val in indata.iter_mut() {
                *val = Complex::new(rng.random::<f32>(), rng.random::<f32>());
            }
            indata[0].im = 0.0;
            if length % 2 == 0 {
                indata[length / 2].im = 0.0;
            }
            for (val_long, val) in rustfft_check
                .iter_mut()
                .take(c2r.complex_len())
                .zip(indata.iter())
            {
                *val_long = *val;
            }
            for (val_long, val) in rustfft_check
                .iter_mut()
                .rev()
                .take(length / 2)
                .zip(indata.iter().skip(1))
            {
                *val_long = val.conj();
            }
            let mut fft_planner = FftPlanner::<f32>::new();
            let fft = fft_planner.plan_fft_inverse(length);

            c2r.process(&mut indata, &mut out_a).unwrap();
            fft.process(&mut rustfft_check);

            let check_real = rustfft_check.iter().map(|val| val.re).collect::<Vec<f32>>();
            let maxdiff = compare_scalars(&out_a, &check_real);
            assert!(
                maxdiff < 5.0e-4,
                "Length: {}, too large error: {}",
                length,
                maxdiff
            );
        }
    }

    // Test that ComplexToReal returns the right errors
    #[test]
    fn complex_to_real_errors_even() {
        let length = 100;
        let mut real_planner = RealFftPlanner::<f64>::new();
        let c2r = real_planner.plan_fft_inverse(length);
        let mut out_a = c2r.make_output_vec();
        let mut indata = c2r.make_input_vec();
        let mut rng = rand::rng();

        // Make some valid data
        for val in indata.iter_mut() {
            *val = Complex::new(rng.random::<f64>(), rng.random::<f64>());
        }
        indata[0].im = 0.0;
        indata[50].im = 0.0;
        // this should be ok
        assert!(c2r.process(&mut indata, &mut out_a).is_ok());

        // Make some invalid data, first point invalid
        for val in indata.iter_mut() {
            *val = Complex::new(rng.random::<f64>(), rng.random::<f64>());
        }
        indata[50].im = 0.0;
        let res = c2r.process(&mut indata, &mut out_a);
        assert!(res.is_err());
        assert!(matches!(res, Err(FftError::InputValues(true, false))));

        // Make some invalid data, last point invalid
        for val in indata.iter_mut() {
            *val = Complex::new(rng.random::<f64>(), rng.random::<f64>());
        }
        indata[0].im = 0.0;
        let res = c2r.process(&mut indata, &mut out_a);
        assert!(res.is_err());
        assert!(matches!(res, Err(FftError::InputValues(false, true))));
    }

    // Test that ComplexToReal returns the right errors
    #[test]
    fn complex_to_real_errors_odd() {
        let length = 101;
        let mut real_planner = RealFftPlanner::<f64>::new();
        let c2r = real_planner.plan_fft_inverse(length);
        let mut out_a = c2r.make_output_vec();
        let mut indata = c2r.make_input_vec();
        let mut rng = rand::rng();

        // Make some valid data
        for val in indata.iter_mut() {
            *val = Complex::new(rng.random::<f64>(), rng.random::<f64>());
        }
        indata[0].im = 0.0;
        // this should be ok
        assert!(c2r.process(&mut indata, &mut out_a).is_ok());

        // Make some invalid data, first point invalid
        for val in indata.iter_mut() {
            *val = Complex::new(rng.random::<f64>(), rng.random::<f64>());
        }
        let res = c2r.process(&mut indata, &mut out_a);
        assert!(res.is_err());
        assert!(matches!(res, Err(FftError::InputValues(true, false))));
    }

    // Compare RealToComplex with standard FFT
    #[test]
    fn real_to_complex_64() {
        for length in 1..1000 {
            let mut real_planner = RealFftPlanner::<f64>::new();
            let r2c = real_planner.plan_fft_forward(length);
            let mut out_a = r2c.make_output_vec();
            let mut indata = r2c.make_input_vec();
            let mut rng = rand::rng();
            for val in indata.iter_mut() {
                *val = rng.random::<f64>();
            }
            let mut rustfft_check = indata
                .iter()
                .map(Complex::from)
                .collect::<Vec<Complex<f64>>>();
            let mut fft_planner = FftPlanner::<f64>::new();
            let fft = fft_planner.plan_fft_forward(length);
            fft.process(&mut rustfft_check);
            r2c.process(&mut indata, &mut out_a).unwrap();
            assert_eq!(out_a[0].im, 0.0, "First imaginary component must be zero");
            if length % 2 == 0 {
                assert_eq!(
                    out_a.last().unwrap().im,
                    0.0,
                    "Last imaginary component for even lengths must be zero"
                );
            }
            let maxdiff = compare_complex(&out_a, &rustfft_check[0..r2c.complex_len()]);
            assert!(
                maxdiff < 1.0e-9,
                "Length: {}, too large error: {}",
                length,
                maxdiff
            );
        }
    }

    // Compare RealToComplex with standard FFT
    #[test]
    fn real_to_complex_32() {
        for length in 1..1000 {
            let mut real_planner = RealFftPlanner::<f32>::new();
            let r2c = real_planner.plan_fft_forward(length);
            let mut out_a = r2c.make_output_vec();
            let mut indata = r2c.make_input_vec();
            let mut rng = rand::rng();
            for val in indata.iter_mut() {
                *val = rng.random::<f32>();
            }
            let mut rustfft_check = indata
                .iter()
                .map(Complex::from)
                .collect::<Vec<Complex<f32>>>();
            let mut fft_planner = FftPlanner::<f32>::new();
            let fft = fft_planner.plan_fft_forward(length);
            fft.process(&mut rustfft_check);
            r2c.process(&mut indata, &mut out_a).unwrap();
            assert_eq!(out_a[0].im, 0.0, "First imaginary component must be zero");
            if length % 2 == 0 {
                assert_eq!(
                    out_a.last().unwrap().im,
                    0.0,
                    "Last imaginary component for even lengths must be zero"
                );
            }
            let maxdiff = compare_complex(&out_a, &rustfft_check[0..r2c.complex_len()]);
            assert!(
                maxdiff < 5.0e-4,
                "Length: {}, too large error: {}",
                length,
                maxdiff
            );
        }
    }

    // Check that the ? operator works on the custom errors. No need to run, just needs to compile.
    #[allow(dead_code)]
    fn test_error() -> Result<(), Box<dyn Error>> {
        let mut real_planner = RealFftPlanner::<f64>::new();
        let r2c = real_planner.plan_fft_forward(100);
        let mut out_a = r2c.make_output_vec();
        let mut indata = r2c.make_input_vec();
        r2c.process(&mut indata, &mut out_a)?;
        Ok(())
    }
}
