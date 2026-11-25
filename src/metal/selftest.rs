//! GPU self-tests for the test suite: the 64-bit atomic minimum and the skeletoniser's float rules
//! (correctly rounded square roots, divisions and reciprocals, and the PDRF's products and fused multiply-adds)

use color_eyre::eyre::{Result, bail, ensure};

use super::gpu::{Arg, Gpu};
use crate::skel::{PenaltyForm, pdrf, radius};

pub fn run(gpu: &Gpu, name: &str) -> Result<String> {
    match name {
        "atomic64" => atomic64(gpu),
        "float_rules" => float_rules(gpu),
        _ => bail!("no GPU self-test named {name:?}"),
    }
}

/// Splitmix64, for reproducible inputs
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn atomic64(gpu: &Gpu) -> Result<String> {
    let n = 1 << 20;
    let mut rng = Rng(7);
    // keys shaped like the packed (cost << 32 | index) of the path search: equal high words, so the low word decides
    let input: Vec<u64> = (0..n).map(|_| (rng.next() & 0xffff_0000_ffff_ffff) | 0x3f80_0000_0000_0000).collect();
    let mut want = [u64::MAX; 4];
    for (i, &v) in input.iter().enumerate() {
        want[i % 4] = want[i % 4].min(v);
    }
    let inp = gpu.upload(&input)?;
    let mut runs = Vec::new();
    for _ in 0..3 {
        let out = gpu.upload(&[u64::MAX; 4])?;
        gpu.dispatch_1d("a64_probe", n, 256, &[(&out).into(), (&inp).into()])?;
        gpu.synchronize()?;
        // SAFETY: completed
        let got: [u64; 4] = unsafe { out.host::<u64>()[..4].try_into().expect("4 words") };
        runs.push(got);
    }
    ensure!(runs.iter().all(|r| *r == want), "64-bit atomic min gave {runs:x?}, want {want:x?}");
    Ok(format!("atomic64: atomic_min_explicit on atomic_ulong compiles and is exact over {n} keys, 3 runs"))
}

fn float_rules(gpu: &Gpu) -> Result<String> {
    // every n below 2^24, then a spread up to 2^32
    let mut ns: Vec<u32> = (0..1u32 << 24).collect();
    let mut rng = Rng(11);
    ns.extend((0..1 << 20).map(|_| rng.next() as u32));
    ns.extend([2 * 2 + 7947 * 7947, u32::MAX, u32::MAX - 1, (1 << 31) + 1]);
    let nb = gpu.upload(&ns)?;
    let out = gpu.alloc(ns.len() * 4)?;
    gpu.dispatch_1d("sk_test_sqrt", ns.len(), 256, &[(&nb).into(), (&out).into()])?;

    // divisions shaped like the PDRF's: radii over their maximum, distances over theirs, and random positives
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..1 << 22 {
        let r = rng.next();
        let (a, b) = ((r & 0xffff) as u32 + 1, ((r >> 16) & 0xffff) as u32 + 1);
        let (lo, hi) = (a.min(b), a.max(b));
        xs.push(2.0 * (f64::from(lo).sqrt() as f32));
        ys.push(2.0 * (f64::from(hi).sqrt() as f32));
        let x = f32::from_bits(0x3000_0000 + (rng.next() as u32 % 0x1800_0000));
        let y = f32::from_bits(0x3000_0000 + (rng.next() as u32 % 0x1800_0000));
        xs.push(x);
        ys.push(y);
    }
    // the PDRF's reciprocals: every significand, at a spread of exponents
    for m in 0..1u32 << 23 {
        let e = (rng.next() % 16) as i32 - 2;
        xs.push(1.0);
        ys.push(f32::from_bits(0x3f80_0000 | m) * 2f32.powi(e));
    }
    let xb = gpu.upload(&xs)?;
    let yb = gpu.upload(&ys)?;
    let qb = gpu.alloc(xs.len() * 4)?;
    gpu.dispatch_1d("sk_test_div", xs.len(), 256, &[(&xb).into(), (&yb).into(), (&qb).into()])?;
    gpu.synchronize()?;

    // SAFETY: completed
    let got = unsafe { out.host::<u32>() };
    let bad_sqrt: Vec<u32> = ns
        .iter()
        .zip(got)
        .filter(|&(&n, &g)| (f64::from(n).sqrt() as f32).to_bits() != g)
        .map(|(&n, _)| n)
        .take(8)
        .collect();
    // SAFETY: completed
    let q = unsafe { qb.host::<u32>() };
    let bad_div: Vec<(f32, f32)> =
        (0..xs.len()).filter(|&i| (xs[i] / ys[i]).to_bits() != q[i]).map(|i| (xs[i], ys[i])).take(8).collect();
    ensure!(
        bad_sqrt.is_empty() && bad_div.is_empty(),
        "float rules differ from the CPU: sqrt at {bad_sqrt:?}, div at {bad_div:?}"
    );
    let n_pdrf = pdrf_rules(gpu, &mut rng)?;
    Ok(format!(
        "float_rules: {} square roots, {} divisions and {n_pdrf} PDRF evaluations match the CPU bit for bit",
        ns.len(),
        xs.len()
    ))
}

/// The PDRF on fields shaped like a piece's: radii of squared distances under their maximum, distances under theirs,
/// in both penalty forms, with and without the distance term
fn pdrf_rules(gpu: &Gpu, rng: &mut Rng) -> Result<usize> {
    let n = 1 << 21;
    let mut inp = Vec::with_capacity(n);
    for _ in 0..n {
        let r = rng.next();
        let (a, b) = ((r & 0xffff) as u32 + 1, ((r >> 16) & 0xffff) as u32 + 1);
        let max_b = radius(a.max(b), 2.0);
        let max_d = f32::from_bits(0x3f80_0000 + ((r >> 32) as u32 % 0x0500_0000));
        let d = max_d * f32::from_bits(0x3f00_0000 | (rng.next() as u32 & 0x7f_ffff));
        inp.push([radius(a.min(b), 2.0), 1.0 / max_b, d.min(max_d), 1.0 / max_d]);
    }
    let ib = gpu.upload(&inp)?;
    let mut checked = 0;
    for (form, daf) in [(PenaltyForm::Squared, true), (PenaltyForm::Unsquared, true), (PenaltyForm::Squared, false)] {
        let flags = if daf { 8 } else { 0 } | if form == PenaltyForm::Unsquared { 128 } else { 0 };
        let out = gpu.alloc(n * 4)?;
        gpu.dispatch_1d("sk_test_pdrf", n, 256, &[(&ib).into(), (&out).into(), Arg::U32(flags), Arg::U32(0)])?;
        gpu.synchronize()?;
        // SAFETY: completed
        let got = unsafe { &out.host::<u32>()[..n] };
        let bad: Vec<[f32; 4]> = (0..n)
            .filter(|&i| {
                let [b, rb, d, rd] = inp[i];
                pdrf(b, rb, d, rd, form, daf).to_bits() != got[i]
            })
            .map(|i| inp[i])
            .take(4)
            .collect();
        ensure!(bad.is_empty(), "PDRF ({form:?}, daf {daf}) differs from the CPU at {bad:?}");
        checked += n;
    }
    Ok(checked)
}
