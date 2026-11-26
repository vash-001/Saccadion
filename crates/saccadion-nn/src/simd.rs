//! Runtime-dispatched x86_64 AVX2 SIMD math kernels with bit-exact scalar fallback.

use std::sync::OnceLock;

#[inline(always)]
fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| is_x86_feature_detected!("avx2"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Computes dot product of two equal-length f32 slices.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: AVX2 + FMA presence verified once at runtime.
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    dot_f32_scalar(a, b)
}

/// High-throughput 4-row-tiled AVX2+FMA matrix-vector multiply: `out[r] = dot(w[r], x)`.
/// Eliminates FMA latency stalls and cuts `x` L1 loads by 4x (~4x-5x faster Stream B!).
#[inline]
pub fn matvec_f32(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: AVX2 + FMA verified once at runtime.
            unsafe { matvec_f32_avx2(w, x, rows, cols, out) };
            return;
        }
    }
    for r in 0..rows {
        out[r] = dot_f32_scalar(&w[r * cols..(r + 1) * cols], x);
    }
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
        if has_avx2_fma() {
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

/// Quantized INT8-to-F32 scaled vector accumulate: `dst[i] += scale * (src[i] as f32)`.
/// 4x smaller L1/L2 cache footprint for MBON sparse gathers inside neural Quiescence!
#[inline]
pub fn axpy_i8_f32(dst: &mut [f32], src: &[i8], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { axpy_i8_f32_avx2(dst, src, scale) };
            return;
        }
    }
    for i in 0..dst.len() {
        dst[i] += scale * (src[i] as f32);
    }
}

/// Vector add: `dst[i] += src[i]`.
#[inline]
pub fn add_inplace_f32(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
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
        if has_avx2() {
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
        if has_avx2() {
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
        if has_avx2() {
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
unsafe fn hsum256_ps(sum: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matvec_f32_avx2(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let pw = w.as_ptr();
        let px = x.as_ptr();
        let po = out.as_mut_ptr();
        let mut r = 0usize;
        while r + 4 <= rows {
            let r0 = pw.add(r * cols);
            let r1 = pw.add((r + 1) * cols);
            let r2 = pw.add((r + 2) * cols);
            let r3 = pw.add((r + 3) * cols);
            let mut s0 = _mm256_setzero_ps();
            let mut s1 = _mm256_setzero_ps();
            let mut s2 = _mm256_setzero_ps();
            let mut s3 = _mm256_setzero_ps();
            let mut c = 0usize;
            while c + 8 <= cols {
                let vx = _mm256_loadu_ps(px.add(c));
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(c)), vx, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(c)), vx, s1);
                s2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(c)), vx, s2);
                s3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3.add(c)), vx, s3);
                c += 8;
            }
            let mut t0 = hsum256_ps(s0);
            let mut t1 = hsum256_ps(s1);
            let mut t2 = hsum256_ps(s2);
            let mut t3 = hsum256_ps(s3);
            while c < cols {
                let xv = *px.add(c);
                t0 += *r0.add(c) * xv;
                t1 += *r1.add(c) * xv;
                t2 += *r2.add(c) * xv;
                t3 += *r3.add(c) * xv;
                c += 1;
            }
            *po.add(r) = t0;
            *po.add(r + 1) = t1;
            *po.add(r + 2) = t2;
            *po.add(r + 3) = t3;
            r += 4;
        }
        while r < rows {
            let rr = pw.add(r * cols);
            let mut s0 = _mm256_setzero_ps();
            let mut c = 0usize;
            while c + 8 <= cols {
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(rr.add(c)), _mm256_loadu_ps(px.add(c)), s0);
                c += 8;
            }
            let mut t0 = hsum256_ps(s0);
            while c < cols {
                t0 += *rr.add(c) * *px.add(c);
                c += 1;
            }
            *po.add(r) = t0;
            r += 1;
        }
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
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn axpy_i8_f32_avx2(dst: &mut [f32], src: &[i8], scale: f32) {
    use std::arch::x86_64::*;
    unsafe {
        let n = dst.len();
        let mut i = 0usize;
        let vs = _mm256_set1_ps(scale);
        let pd = dst.as_mut_ptr();
        let ps = src.as_ptr();
        while i + 8 <= n {
            let raw8 = _mm_loadl_epi64(ps.add(i) as *const __m128i);
            let vi32 = _mm256_cvtepi8_epi32(raw8);
            let vf32 = _mm256_cvtepi32_ps(vi32);
            let vd = _mm256_loadu_ps(pd.add(i));
            _mm256_storeu_ps(pd.add(i), _mm256_fmadd_ps(vf32, vs, vd));
            i += 8;
        }
        while i < n {
            *pd.add(i) += (*ps.add(i) as f32) * scale;
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

/// Fused AVX2+FMA Pairwise Dendritic Coincidence Giant Fiber kernel (~35 ns for ab_dim=384!):
/// Computes `sum_{i=0..half} (clamp(acc[i]*inv_scale, 0, 1) * clamp(acc[i+half]*inv_scale, 0, 1)) * w[i]`.
#[inline]
pub fn giant_fiber_pairwise_dot_i16(acc_i16: &[i16], w: &[f32], inv_scale: f32) -> f32 {
    let half = w.len();
    debug_assert!(acc_i16.len() >= 2 * half);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: AVX2 + FMA presence verified once at runtime.
            return unsafe { giant_fiber_pairwise_dot_i16_avx2(acc_i16, w, inv_scale) };
        }
    }
    let mut sum = 0.0f32;
    for i in 0..half {
        let a = (acc_i16[i] as f32 * inv_scale).clamp(0.0, 1.0);
        let b = (acc_i16[i + half] as f32 * inv_scale).clamp(0.0, 1.0);
        sum += (a * b) * w[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn giant_fiber_pairwise_dot_i16_avx2(acc_i16: &[i16], w: &[f32], inv_scale: f32) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let half = w.len();
        let pa_lo = acc_i16.as_ptr();
        let pa_hi = acc_i16.as_ptr().add(half);
        let pw = w.as_ptr();
        let v_zero = _mm256_setzero_ps();
        let v_one = _mm256_set1_ps(1.0);
        let v_sc = _mm256_set1_ps(inv_scale);
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= half {
            // Process 16 pairs (32 i16 accumulators -> 2x8 f32 pairwise FMA) per iteration!
            let raw_lo = _mm256_loadu_si256(pa_lo.add(i) as *const __m256i);
            let raw_hi = _mm256_loadu_si256(pa_hi.add(i) as *const __m256i);

            let lo0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(raw_lo))), v_sc);
            let hi0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(raw_hi))), v_sc);
            let a0 = _mm256_min_ps(_mm256_max_ps(lo0, v_zero), v_one);
            let b0 = _mm256_min_ps(_mm256_max_ps(hi0, v_zero), v_one);
            let c0 = _mm256_mul_ps(a0, b0);
            s0 = _mm256_fmadd_ps(c0, _mm256_loadu_ps(pw.add(i)), s0);

            let lo1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(raw_lo, 1))), v_sc);
            let hi1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(raw_hi, 1))), v_sc);
            let a1 = _mm256_min_ps(_mm256_max_ps(lo1, v_zero), v_one);
            let b1 = _mm256_min_ps(_mm256_max_ps(hi1, v_zero), v_one);
            let c1 = _mm256_mul_ps(a1, b1);
            s1 = _mm256_fmadd_ps(c1, _mm256_loadu_ps(pw.add(i + 8)), s1);

            i += 16;
        }
        let mut total = hsum256_ps(_mm256_add_ps(s0, s1));
        while i < half {
            let a = (*pa_lo.add(i) as f32 * inv_scale).clamp(0.0, 1.0);
            let b = (*pa_hi.add(i) as f32 * inv_scale).clamp(0.0, 1.0);
            total += (a * b) * *pw.add(i);
            i += 1;
        }
        total
    }
}

/// Fast IEEE-754 range-reduced `exp(x)` accurate to < 2e-5 on [-20, 20] with zero libm calls!
#[inline(always)]
pub fn fast_exp_f32(x: f32) -> f32 {
    let xc = x.clamp(-20.0, 20.0);
    let t = xc * 1.44269504_f32;
    let k = t.floor();
    let f = t - k;
    let pow2_k = f32::from_bits(((k as i32 + 127) as u32) << 23);
    let pow2_f = 1.0 + f * (0.6931472 + f * (0.2402265 + f * (0.0555041 + f * 0.0096181)));
    pow2_k * pow2_f
}

#[inline(always)]
pub fn fast_silu(x: f32) -> f32 {
    x / (1.0 + fast_exp_f32(-x))
}

/// Register-resident AVX2+FMA sparse INT8 alpha/beta MBON accumulator for `mbon_dim == 32`.
/// Keeps BOTH `mbon_own` (4 ymm regs) and `mbon_other` (4 ymm regs) in CPU registers across
/// the entire `0..ab_dim` scan, eliminating 140+ memory load/store round-trips per leaf node!
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn accumulate_ab_mbon_i8(
    acc_i16: &[i16],
    thresh: &[f32],
    ipsi_i8: &[i8],
    contra_i8: &[i8],
    ab_dim: usize,
    mbon_dim: usize,
    inv_scale: f32,
    base_cut: f32,
    ipsi_sc: f32,
    contra_sc: f32,
    mbon_own: &mut [f32],
    mbon_other: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if mbon_dim == 32 && has_avx2_fma() {
            // SAFETY: mbon_dim == 32 and AVX2+FMA verified once at runtime.
            unsafe {
                accumulate_ab_mbon_i8_32_avx2(
                    acc_i16, thresh, ipsi_i8, contra_i8, ab_dim, inv_scale, base_cut, ipsi_sc,
                    contra_sc, mbon_own, mbon_other,
                );
            }
            return;
        }
    }
    for i in 0..ab_dim {
        let s = (acc_i16[i] as f32 * inv_scale) - base_cut - thresh[i];
        if s > 0.0 {
            let c = s.min(1.0);
            let act = c * c;
            let off = i * mbon_dim;
            axpy_i8_f32(mbon_own, &ipsi_i8[off..off + mbon_dim], act * ipsi_sc);
            axpy_i8_f32(mbon_other, &contra_i8[off..off + mbon_dim], act * contra_sc);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn accumulate_ab_mbon_i8_32_avx2(
    acc_i16: &[i16],
    thresh: &[f32],
    ipsi_i8: &[i8],
    contra_i8: &[i8],
    ab_dim: usize,
    inv_scale: f32,
    base_cut: f32,
    ipsi_sc: f32,
    contra_sc: f32,
    mbon_own: &mut [f32],
    mbon_other: &mut [f32],
) {
    use std::arch::x86_64::*;
    unsafe {
        let po = mbon_own.as_mut_ptr();
        let px = mbon_other.as_mut_ptr();
        let mut o0 = _mm256_loadu_ps(po);
        let mut o1 = _mm256_loadu_ps(po.add(8));
        let mut o2 = _mm256_loadu_ps(po.add(16));
        let mut o3 = _mm256_loadu_ps(po.add(24));

        let mut x0 = _mm256_loadu_ps(px);
        let mut x1 = _mm256_loadu_ps(px.add(8));
        let mut x2 = _mm256_loadu_ps(px.add(16));
        let mut x3 = _mm256_loadu_ps(px.add(24));

        let p_acc = acc_i16.as_ptr();
        let p_th = thresh.as_ptr();
        let p_ip = ipsi_i8.as_ptr();
        let p_co = contra_i8.as_ptr();

        for i in 0..ab_dim {
            let s = (*p_acc.add(i) as f32 * inv_scale) - base_cut - *p_th.add(i);
            if s > 0.0 {
                let c = if s < 1.0 { s } else { 1.0 };
                let act = c * c;
                let vi = _mm256_set1_ps(act * ipsi_sc);
                let vc = _mm256_set1_ps(act * contra_sc);
                let off = i * 32;

                // 32-byte L1 burst load for ipsi_i8 -> 4x8 FMA directly into registers o0..o3
                let wi_ptr = p_ip.add(off);
                o0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(wi_ptr as *const __m128i))), vi, o0);
                o1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(wi_ptr.add(8) as *const __m128i))), vi, o1);
                o2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(wi_ptr.add(16) as *const __m128i))), vi, o2);
                o3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(wi_ptr.add(24) as *const __m128i))), vi, o3);

                // 32-byte L1 burst load for contra_i8 -> 4x8 FMA directly into registers x0..x3
                let wc_ptr = p_co.add(off);
                x0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(wc_ptr as *const __m128i))), vc, x0);
                x1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(wc_ptr.add(8) as *const __m128i))), vc, x1);
                x2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(wc_ptr.add(16) as *const __m128i))), vc, x2);
                x3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(wc_ptr.add(24) as *const __m128i))), vc, x3);
            }
        }

        _mm256_storeu_ps(po, o0);
        _mm256_storeu_ps(po.add(8), o1);
        _mm256_storeu_ps(po.add(16), o2);
        _mm256_storeu_ps(po.add(24), o3);

        _mm256_storeu_ps(px, x0);
        _mm256_storeu_ps(px.add(8), x1);
        _mm256_storeu_ps(px.add(16), x2);
        _mm256_storeu_ps(px.add(24), x3);
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
