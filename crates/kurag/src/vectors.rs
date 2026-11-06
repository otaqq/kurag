//! The vectors of one namespace in memory, 8-bit: each component of a unit vector is stored as round(x * 127),
//! a quarter of float32. Search is exact (every vector compared); an ANN index comes when measurements call for one
//! (docs/design.md, "Choosing the ANN index"). The float16 copy in the database is the truth; this is rebuilt from it.

use std::collections::HashMap;

pub struct Vectors {
    dims: usize,
    data: Vec<i8>,
    vids: Vec<i64>,
    pos: HashMap<i64, usize>,
}

fn quantize(x: f32) -> i8 {
    (x * 127.0).round().clamp(-127.0, 127.0) as i8
}

impl Vectors {
    pub fn new(dims: usize) -> Vectors {
        Vectors { dims, data: Vec::new(), vids: Vec::new(), pos: HashMap::new() }
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn len(&self) -> usize {
        self.vids.len()
    }

    /// Adds (or replaces) the vector of chunk `vid`.
    pub fn add(&mut self, vid: i64, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dims);
        self.remove(vid);
        self.pos.insert(vid, self.vids.len());
        self.vids.push(vid);
        self.data.extend(v.iter().map(|&x| quantize(x)));
    }

    /// Removes the vector of chunk `vid` (the last one takes its place).
    pub fn remove(&mut self, vid: i64) {
        let Some(i) = self.pos.remove(&vid) else { return };
        let last = self.vids.len() - 1;
        if i != last {
            let moved = self.vids[last];
            self.vids[i] = moved;
            self.pos.insert(moved, i);
            let (head, tail) = self.data.split_at_mut(last * self.dims);
            head[i * self.dims..(i + 1) * self.dims].copy_from_slice(&tail[..self.dims]);
        }
        self.vids.pop();
        self.data.truncate(last * self.dims);
    }

    fn score_at(&self, q: &[f32], i: usize) -> f32 {
        let row = &self.data[i * self.dims..(i + 1) * self.dims];
        q.iter().zip(row).map(|(a, &b)| a * b as f32).sum::<f32>() / 127.0
    }

    /// Cosine similarity of `q` to chunk `vid`.
    pub fn similarity(&self, q: &[f32], vid: i64) -> Option<f32> {
        self.pos.get(&vid).map(|&i| self.score_at(q, i))
    }

    /// The `k` chunks closest to `q`: (vid, cosine similarity), best first.
    pub fn nearest(&self, q: &[f32], k: usize) -> Vec<(i64, f32)> {
        if k == 0 || q.len() != self.dims {
            return Vec::new();
        }
        let mut all: Vec<(i64, f32)> = (0..self.vids.len()).map(|i| (self.vids[i], self.score_at(q, i))).collect();
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

    #[test]
    fn add_remove_nearest() {
        let mut v = Vectors::new(2);
        v.add(1, &[1.0, 0.0]);
        v.add(2, &[0.0, 1.0]);
        v.add(3, &[0.6, 0.8]);
        assert_eq!(v.nearest(&[1.0, 0.0], 2).iter().map(|x| x.0).collect::<Vec<_>>(), [1, 3]);
        v.remove(1);
        assert_eq!(v.len(), 2);
        assert_eq!(v.nearest(&[1.0, 0.0], 1)[0].0, 3);
        assert!((v.similarity(&[0.0, 1.0], 2).unwrap() - 1.0).abs() < 0.01);
        v.add(3, &[1.0, 0.0]); // replaced
        assert_eq!(v.len(), 2);
        assert!((v.similarity(&[1.0, 0.0], 3).unwrap() - 1.0).abs() < 0.01);
    }
}
