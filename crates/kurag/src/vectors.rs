//! The vectors of one namespace in memory, 8-bit with a range per dimension (the idea of faiss's SQ8): each
//! dimension's minimum and maximum are learned from the vectors, and a component is stored as one of 256 steps
//! between them, a quarter of float32. A fixed scale for all dimensions (x 127) left most steps unused, since the
//! components of a unit vector with hundreds of dimensions are small, and lost recall (docs/architecture.md).
//!
//! Search is exact (every vector compared); an ANN index comes when measurements call for one. The float16 copies
//! in the database are the truth: the ranges are learned again from them when the namespace is opened, on
//! `/rebuild`, and when the number of vectors has doubled since (values outside the range are clamped meanwhile).

use std::collections::HashMap;

pub struct Vectors {
    dims: usize,
    min: Vec<f32>,
    step: Vec<f32>,
    codes: Vec<u8>,
    vids: Vec<i64>,
    pos: HashMap<i64, usize>,
    trained_on: usize,
}

/// Minimum and maximum of each dimension, gathered vector by vector.
pub struct Range {
    pub min: Vec<f32>,
    pub max: Vec<f32>,
    pub count: usize,
}

impl Range {
    pub fn new(dims: usize) -> Range {
        Range { min: vec![f32::INFINITY; dims], max: vec![f32::NEG_INFINITY; dims], count: 0 }
    }

    pub fn add(&mut self, v: &[f32]) {
        for (d, &x) in v.iter().enumerate() {
            self.min[d] = self.min[d].min(x);
            self.max[d] = self.max[d].max(x);
        }
        self.count += 1;
    }
}

/// The query prepared for one index: q x step per dimension, and the part that does not depend on the codes.
struct Prepared {
    weights: Vec<f32>,
    base: f32,
}

impl Vectors {
    /// An empty index with the ranges learned from `range`.
    pub fn new(range: &Range) -> Vectors {
        let dims = range.min.len();
        let (mut min, mut step) = (vec![0f32; dims], vec![0f32; dims]);
        for d in 0..dims {
            let (lo, hi) = if range.count == 0 { (-1.0, 1.0) } else { (range.min[d], range.max[d]) };
            min[d] = lo;
            step[d] = if hi > lo { (hi - lo) / 255.0 } else { f32::EPSILON };
        }
        Vectors { dims, min, step, codes: Vec::new(), vids: Vec::new(), pos: HashMap::new(), trained_on: range.count }
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn len(&self) -> usize {
        self.vids.len()
    }

    /// How many vectors the ranges were learned from.
    pub fn trained_on(&self) -> usize {
        self.trained_on
    }

    /// Adds (or replaces) the vector of chunk `vid`.
    pub fn add(&mut self, vid: i64, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dims);
        self.remove(vid);
        self.pos.insert(vid, self.vids.len());
        self.vids.push(vid);
        for (d, &x) in v.iter().enumerate() {
            self.codes.push(((x - self.min[d]) / self.step[d]).round().clamp(0.0, 255.0) as u8);
        }
    }

    /// Removes the vector of chunk `vid` (the last one takes its place).
    pub fn remove(&mut self, vid: i64) {
        let Some(i) = self.pos.remove(&vid) else { return };
        let last = self.vids.len() - 1;
        if i != last {
            let moved = self.vids[last];
            self.vids[i] = moved;
            self.pos.insert(moved, i);
            let (head, tail) = self.codes.split_at_mut(last * self.dims);
            head[i * self.dims..(i + 1) * self.dims].copy_from_slice(&tail[..self.dims]);
        }
        self.vids.pop();
        self.codes.truncate(last * self.dims);
    }

    fn prepare(&self, q: &[f32]) -> Prepared {
        Prepared {
            weights: q.iter().zip(&self.step).map(|(a, s)| a * s).collect(),
            base: q.iter().zip(&self.min).map(|(a, m)| a * m).sum(),
        }
    }

    /// x_d = min_d + step_d * code_d, so q . x = sum(q_d * min_d) + sum(q_d * step_d * code_d)
    fn score_at(&self, p: &Prepared, i: usize) -> f32 {
        let row = &self.codes[i * self.dims..(i + 1) * self.dims];
        p.base + p.weights.iter().zip(row).map(|(w, &c)| w * c as f32).sum::<f32>()
    }

    /// Cosine similarity of `q` to chunk `vid`.
    pub fn similarity(&self, q: &[f32], vid: i64) -> Option<f32> {
        if q.len() != self.dims {
            return None;
        }
        self.pos.get(&vid).map(|&i| self.score_at(&self.prepare(q), i))
    }

    /// The `k` chunks closest to `q`: (vid, cosine similarity), best first.
    pub fn nearest(&self, q: &[f32], k: usize) -> Vec<(i64, f32)> {
        if k == 0 || q.len() != self.dims {
            return Vec::new();
        }
        let p = self.prepare(q);
        let mut all: Vec<(i64, f32)> = (0..self.vids.len()).map(|i| (self.vids[i], self.score_at(&p, i))).collect();
        let by_score = |a: &(i64, f32), b: &(i64, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        if all.len() > k {
            all.select_nth_unstable_by(k - 1, by_score);
            all.truncate(k);
        }
        all.sort_by(by_score);
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(rows: &[(i64, Vec<f32>)]) -> Vectors {
        let mut r = Range::new(rows[0].1.len());
        rows.iter().for_each(|(_, v)| r.add(v));
        let mut v = Vectors::new(&r);
        rows.iter().for_each(|(vid, x)| v.add(*vid, x));
        v
    }

    #[test]
    fn add_remove_nearest() {
        let mut v = index(&[(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0]), (3, vec![0.6, 0.8])]);
        assert_eq!(v.trained_on(), 3);
        assert_eq!(v.nearest(&[1.0, 0.0], 2).iter().map(|x| x.0).collect::<Vec<_>>(), [1, 3]);
        v.remove(1);
        assert_eq!(v.len(), 2);
        assert_eq!(v.nearest(&[1.0, 0.0], 1)[0].0, 3);
        assert!((v.similarity(&[0.0, 1.0], 2).unwrap() - 1.0).abs() < 0.01);
        v.add(3, &[1.0, 0.0]); // replaced
        assert_eq!(v.len(), 2);
        assert!((v.similarity(&[1.0, 0.0], 3).unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn small_components_keep_precision() {
        // components around +-0.04, as in a 768-dimension unit vector: the error stays far below the gaps
        let rows: Vec<(i64, Vec<f32>)> =
            (0..50).map(|i| (i, (0..8).map(|d| (((i * 7 + d * 3) % 17) as f32 - 8.0) / 200.0).collect())).collect();
        let v = index(&rows);
        let q = &rows[10].1;
        for (vid, x) in &rows {
            let exact: f32 = q.iter().zip(x).map(|(a, b)| a * b).sum();
            assert!((v.similarity(q, *vid).unwrap() - exact).abs() < 1e-4, "vid {vid}");
        }
    }
}
