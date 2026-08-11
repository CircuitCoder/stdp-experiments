use serde::{Deserialize, Serialize};

/// Connectivity + weight storage for a synapse group.
///
/// All variants store enough structure to iterate both "by pre-neuron"
/// (forward / CSR) and "by post-neuron" (reverse / CSC).
///
/// Indices are **group-local**: pre-neuron 0 is the first neuron of the
/// pre-group, post-neuron 0 is the first of the post-group.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Weight {
    /// Dense connection matrix (num_pre × num_post).
    /// Row-major: `data[pre * num_post + post]`.
    Dense {
        num_pre: usize,
        num_post: usize,
        data: Vec<f32>,
    },

    /// One-to-one connection (num_pre == num_post).
    /// `data[i]` connects pre i → post i.
    OneToOne {
        data: Vec<f32>,
    },
}

impl Weight {
    pub fn num_pre(&self) -> usize {
        match self {
            Weight::Dense { num_pre, .. } => *num_pre,
            Weight::OneToOne { data } => data.len(),
        }
    }

    pub fn num_post(&self) -> usize {
        match self {
            Weight::Dense { num_post, .. } => *num_post,
            Weight::OneToOne { data } => data.len(),
        }
    }

    pub fn num_synapses(&self) -> usize {
        match self {
            Weight::Dense { data, .. } => data.len(),
            Weight::OneToOne { data } => data.len(),
        }
    }

    /// Get the weight for a specific (pre, post) pair.
    #[inline(always)]
    pub fn get(&self, pre: usize, post: usize) -> f32 {
        match self {
            Weight::Dense { num_post, data, .. } => data[pre * num_post + post],
            Weight::OneToOne { data } => {
                if pre == post { data[pre] } else { 0.0 }
            }
        }
    }

    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------

    pub fn new_dense(num_pre: usize, num_post: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), num_pre * num_post);
        Weight::Dense { num_pre, num_post, data }
    }

    pub fn new_one_to_one(data: Vec<f32>) -> Self {
        Weight::OneToOne { data }
    }

    // ---------------------------------------------------------------
    // Bulk accumulation — auto-vectorisable hot path.
    // ---------------------------------------------------------------

    /// `target[post] += weight` for every synapse from `pre_id`.
    #[inline(always)]
    pub fn accumulate_pre(&self, pre_id: usize, target: &mut [f32]) {
        match self {
            Weight::Dense { num_post, data, .. } => {
                let row = &data[pre_id * num_post..(pre_id + 1) * num_post];
                let target = &mut target[..*num_post];
                for j in 0..*num_post {
                    target[j] += row[j];
                }
            }
            Weight::OneToOne { data } => {
                target[pre_id] += data[pre_id];
            }
        }
    }

    // ---------------------------------------------------------------
    // Per-synapse iteration for STDP / normalization.
    // ---------------------------------------------------------------

    /// Iterate all synapses from `pre_id`: yields (post_id, &mut weight).
    #[inline(always)]
    pub fn iter_pre_mut(&mut self, pre_id: usize) -> PreIterMut<'_> {
        match self {
            Weight::Dense { num_post, data, .. } => {
                let np = *num_post;
                let start = pre_id * np;
                PreIterMut::Dense { col: 0, num_post: np, data: &mut data[start..start + np] }
            }
            Weight::OneToOne { data } => {
                PreIterMut::OneToOne { id: pre_id, data: &mut data[pre_id..=pre_id], done: false }
            }
        }
    }

    /// Iterate all synapses into `post_id`: yields (pre_id, &mut weight).
    #[inline(always)]
    pub fn iter_post_mut(&mut self, post_id: usize) -> PostIterMut<'_> {
        match self {
            Weight::Dense { num_pre, num_post, data, .. } => {
                PostIterMut::Dense {
                    row: 0, num_pre: *num_pre, num_post: *num_post,
                    post_id, data: data.as_mut_ptr(),
                }
            }
            Weight::OneToOne { data } => {
                PostIterMut::OneToOne { id: post_id, data: &mut data[post_id..=post_id], done: false }
            }
        }
    }

    /// Read-only iteration of all synapses into `post_id`: yields (pre_id, weight).
    #[inline(always)]
    pub fn iter_post(&self, post_id: usize) -> PostIter<'_> {
        match self {
            Weight::Dense { num_pre, num_post, data, .. } => {
                PostIter::Dense { row: 0, num_pre: *num_pre, num_post: *num_post, post_id, data }
            }
            Weight::OneToOne { data } => {
                PostIter::OneToOne { id: post_id, weight: data[post_id], done: false }
            }
        }
    }

    /// Iterate all weights: calls f(pre, post, weight).
    pub fn for_each_weight(&self, mut f: impl FnMut(usize, usize, f32)) {
        match self {
            Weight::Dense { num_pre, num_post, data } => {
                for i in 0..*num_pre {
                    for j in 0..*num_post {
                        f(i, j, data[i * num_post + j]);
                    }
                }
            }
            Weight::OneToOne { data } => {
                for (i, &w) in data.iter().enumerate() {
                    f(i, i, w);
                }
            }
        }
    }

    /// Mutable iteration over all weights: calls f(pre, post, &mut weight).
    pub fn for_each_weight_mut(&mut self, mut f: impl FnMut(usize, usize, &mut f32)) {
        match self {
            Weight::Dense { num_pre, num_post, data } => {
                let np = *num_post;
                for i in 0..*num_pre {
                    for j in 0..np {
                        f(i, j, &mut data[i * np + j]);
                    }
                }
            }
            Weight::OneToOne { data } => {
                for (i, w) in data.iter_mut().enumerate() {
                    f(i, i, w);
                }
            }
        }
    }

    /// Direct mutable access to the weight data slice (row-major for Dense,
    /// element-indexed for OneToOne).
    pub fn data_mut(&mut self) -> &mut [f32] {
        match self {
            Weight::Dense { data, .. } => data,
            Weight::OneToOne { data } => data,
        }
    }

    pub fn data(&self) -> &[f32] {
        match self {
            Weight::Dense { data, .. } => data,
            Weight::OneToOne { data } => data,
        }
    }
}

// ---------------------------------------------------------------
// Iterator types.
// ---------------------------------------------------------------

pub enum PreIterMut<'a> {
    Dense {
        col: usize,
        num_post: usize,
        data: &'a mut [f32],
    },
    OneToOne {
        id: usize,
        data: &'a mut [f32],
        done: bool,
    },
}

impl<'a> Iterator for PreIterMut<'a> {
    type Item = (usize, &'a mut f32);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PreIterMut::Dense { col, num_post, data } => {
                if *col >= *num_post {
                    return None;
                }
                let c = *col;
                *col += 1;
                // SAFETY: we yield each index exactly once (col advances monotonically).
                let ptr = data.as_mut_ptr();
                Some((c, unsafe { &mut *ptr.add(c) }))
            }
            PreIterMut::OneToOne { id, data, done } => {
                if *done {
                    return None;
                }
                *done = true;
                let ptr = data.as_mut_ptr();
                Some((*id, unsafe { &mut *ptr }))
            }
        }
    }
}

pub enum PostIterMut<'a> {
    Dense {
        row: usize,
        num_pre: usize,
        num_post: usize,
        post_id: usize,
        data: *mut f32,
    },
    OneToOne {
        id: usize,
        data: &'a mut [f32],
        done: bool,
    },
}

impl<'a> Iterator for PostIterMut<'a> {
    type Item = (usize, &'a mut f32);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PostIterMut::Dense { row, num_pre, num_post, post_id, data } => {
                if *row >= *num_pre {
                    return None;
                }
                let r = *row;
                *row += 1;
                let idx = r * *num_post + *post_id;
                // SAFETY: each (row, post_id) pair is visited once.
                Some((r, unsafe { &mut *data.add(idx) }))
            }
            PostIterMut::OneToOne { id, data, done } => {
                if *done {
                    return None;
                }
                *done = true;
                let ptr = data.as_mut_ptr();
                Some((*id, unsafe { &mut *ptr }))
            }
        }
    }
}

pub enum PostIter<'a> {
    Dense {
        row: usize,
        num_pre: usize,
        num_post: usize,
        post_id: usize,
        data: &'a [f32],
    },
    OneToOne {
        id: usize,
        weight: f32,
        done: bool,
    },
}

impl<'a> Iterator for PostIter<'a> {
    type Item = (usize, f32);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PostIter::Dense { row, num_pre, num_post, post_id, data } => {
                if *row >= *num_pre {
                    return None;
                }
                let r = *row;
                *row += 1;
                Some((r, data[r * *num_post + *post_id]))
            }
            PostIter::OneToOne { id, weight, done } => {
                if *done {
                    return None;
                }
                *done = true;
                Some((*id, *weight))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_accumulate_pre() {
        let w = Weight::new_dense(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut target = vec![0.0; 3];
        w.accumulate_pre(0, &mut target);
        assert_eq!(target, vec![1.0, 2.0, 3.0]);
        w.accumulate_pre(1, &mut target);
        assert_eq!(target, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn one_to_one_accumulate_pre() {
        let w = Weight::new_one_to_one(vec![10.0, 20.0, 30.0]);
        let mut target = vec![0.0; 3];
        w.accumulate_pre(1, &mut target);
        assert_eq!(target, vec![0.0, 20.0, 0.0]);
    }

    #[test]
    fn dense_iter_pre_mut() {
        let mut w = Weight::new_dense(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let items: Vec<_> = w.iter_pre_mut(1).map(|(j, wt)| { *wt += 1.0; (j, *wt) }).collect();
        assert_eq!(items, vec![(0, 5.0), (1, 6.0), (2, 7.0)]);
    }

    #[test]
    fn dense_iter_post() {
        let w = Weight::new_dense(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let items: Vec<_> = w.iter_post(1).collect();
        assert_eq!(items, vec![(0, 2.0), (1, 5.0)]);
    }

    #[test]
    fn dense_iter_post_mut() {
        let mut w = Weight::new_dense(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let items: Vec<_> = w.iter_post_mut(1).map(|(i, wt)| { *wt *= 2.0; (i, *wt) }).collect();
        assert_eq!(items, vec![(0, 4.0), (1, 10.0)]);
    }

    #[test]
    fn one_to_one_iter_post_mut() {
        let mut w = Weight::new_one_to_one(vec![10.0, 20.0, 30.0]);
        let items: Vec<_> = w.iter_post_mut(1).map(|(i, wt)| { *wt = 0.0; (i, *wt) }).collect();
        assert_eq!(items, vec![(1, 0.0)]);
        let mut target = vec![0.0; 3];
        w.accumulate_pre(1, &mut target);
        assert_eq!(target, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn dense_for_each_weight() {
        let w = Weight::new_dense(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let mut pairs = Vec::new();
        w.for_each_weight(|pre, post, wt| pairs.push((pre, post, wt)));
        assert_eq!(pairs, vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)]);
    }
}
