//! Runtime-dispatched x86_64 AVX2 SIMD math kernels with bit-exact scalar fallback.

/// Computes dot product of two equal-length f32 slices.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 + FMA presence verified at runtime.
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    dot_f32_scalar(a, b)
}

#[inline]
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Vector accumulate: `dst[i] += scale * src[i]` for all `i`.
#[inline]
pub fn axpy_f32(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 + FMA presence verified at runtime.
            unsafe { axpy_f32_avx2(dst, src, scale) };
            return;
        }
    }
    axpy_f32_scalar(dst, src, scale);
}

#[inline]
pub fn axpy_f32_scalar(dst: &mut [f32], src: &[f32], scale: f32) {
    for i in 0..dst.len() {
        dst[i] += scale * src[i];
    }
}

/// Vector add: `dst[i] += src[i]`.
#[inline]
pub fn add_inplace_f32(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified at runtime.
            unsafe { add_inplace_f32_avx2(dst, src) };
            return;
        }
    }
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

/// Vector sub: `dst[i] -= src[i]`.
#[inline]
pub fn sub_inplace_f32(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified at runtime.
            unsafe { sub_inplace_f32_avx2(dst, src) };
            return;
        }
    }
    for i in 0..dst.len() {
        dst[i] -= src[i];
    }
}

/// Quantized INT16 vector add: `dst[i] = dst[i].saturating_add(src[i])` (16 neurons/AVX2 op!).
#[inline]
pub fn add_inplace_i16(dst: &mut [i16], src: &[i16]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { add_inplace_i16_avx2(dst, src) };
            return;
        }
    }
    for i in 0..dst.len() {
        dst[i] = dst[i].saturating_add(src[i]);
    }
}

/// Quantized INT16 vector sub: `dst[i] = dst[i].saturating_sub(src[i])` (16 neurons/AVX2 op!).
#[inline]
pub fn sub_inplace_i16(dst: &mut [i16], src: &[i16]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sub_inplace_i16_avx2(dst, src) };
            return;
        }
    }
    for i in 0..dst.len() {
        dst[i] = dst[i].saturating_sub(src[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let n = a.len();
        let mut i = 0usize;
        let mut sum = _mm256_setzero_ps();
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        while i + 8 <= n {
            let va = _mm256_loadu_ps(pa.add(i));
            let vb = _mm256_loadu_ps(pb.add(i));
            sum = _mm256_fmadd_ps(va, vb, sum);
            i += 8;
        }
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let s128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let mut total = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        while i < n {
            total += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn axpy_f32_avx2(dst: &mut [f32], src: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    unsafe {
        let n = dst.len();
        let mut i = 0usize;
        let vs = _mm256_set1_ps(scale);
        let pd = dst.as_mut_ptr();
        let ps = src.as_ptr();
        while i + 8 <= n {
            let vd = _mm256_loadu_ps(pd.add(i));
            let vx = _mm256_loadu_ps(ps.add(i));
            let res = _mm256_fmadd_ps(vx, vs, vd);
            _mm256_storeu_ps(pd.add(i), res);
            i += 8;
        }
        while i < n {
            *pd.add(i) += *ps.add(i) * scale;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_inplace_f32_avx2(dst: &mut [f32], src: &[f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let n = dst.len();
        let mut i = 0usize;
        let pd = dst.as_mut_ptr();
        let ps = src.as_ptr();
        while i + 8 <= n {
            let vd = _mm256_loadu_ps(pd.add(i));
            let vx = _mm256_loadu_ps(ps.add(i));
            _mm256_storeu_ps(pd.add(i), _mm256_add_ps(vd, vx));
            i += 8;
        }
        while i < n {
            *pd.add(i) += *ps.add(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub_inplace_f32_avx2(dst: &mut [f32], src: &[f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let n = dst.len();
        let mut i = 0usize;
        let pd = dst.as_mut_ptr();
        let ps = src.as_ptr();
        while i + 8 <= n {
            let vd = _mm256_loadu_ps(pd.add(i));
            let vx = _mm256_loadu_ps(ps.add(i));
            _mm256_storeu_ps(pd.add(i), _mm256_sub_ps(vd, vx));
            i += 8;
        }
        while i < n {
            *pd.add(i) -= *ps.add(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_inplace_i16_avx2(dst: &mut [i16], src: &[i16]) {
    use std::arch::x86_64::*;
    unsafe {
        let n = dst.len();
        let mut i = 0usize;
        let pd = dst.as_mut_ptr();
        let ps = src.as_ptr();
        while i + 16 <= n {
            let vd = _mm256_loadu_si256(pd.add(i) as *const __m256i);
            let vx = _mm256_loadu_si256(ps.add(i) as *const __m256i);
            _mm256_storeu_si256(pd.add(i) as *mut __m256i, _mm256_adds_epi16(vd, vx));
            i += 16;
        }
        while i < n {
            *pd.add(i) = (*pd.add(i)).saturating_add(*ps.add(i));
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub_inplace_i16_avx2(dst: &mut [i16], src: &[i16]) {
    use std::arch::x86_64::*;
    unsafe {
        let n = dst.len();
        let mut i = 0usize;
        let pd = dst.as_mut_ptr();
        let ps = src.as_ptr();
        while i + 16 <= n {
            let vd = _mm256_loadu_si256(pd.add(i) as *const __m256i);
            let vx = _mm256_loadu_si256(ps.add(i) as *const __m256i);
            _mm256_storeu_si256(pd.add(i) as *mut __m256i, _mm256_subs_epi16(vd, vx));
            i += 16;
        }
        while i < n {
            *pd.add(i) = (*pd.add(i)).saturating_sub(*ps.add(i));
            i += 1;
        }
    }
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

pub fn softmax_inplace(v: &mut [f32]) {
    let max_v = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max_v).exp();
        sum += *x;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for x in v.iter_mut() {
        *x *= inv;
    }
}

pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for i in 0..x.len() {
        out[i] = ((x[i] - mean) * inv_std) * gamma[i] + beta[i];
    }
}
