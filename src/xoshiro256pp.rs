/// Xoshiro256++ PRNG (Blackman & Vigna, 2018), seeded from a single u64 via SplitMix64 expansion.
pub struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    pub fn new(seed: u64) -> Self {
        let mut sm_state = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            sm_state = sm_state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniformly random index in `[0, n)` using Lemire's nearly divisionless method.
    fn next_index(&mut self, n: usize) -> usize {
        let n = n as u64;
        let x = self.next_u64();
        let mut m = (x as u128).wrapping_mul(n as u128);
        let mut l = m as u64;
        if l < n {
            let threshold = n.wrapping_neg() % n;
            while l < threshold {
                let x = self.next_u64();
                m = (x as u128).wrapping_mul(n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as usize
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_index(i + 1);
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Xoshiro256PlusPlus;

    #[test]
    fn deterministic_from_same_seed() {
        let mut a = Xoshiro256PlusPlus::new(42);
        let mut b = Xoshiro256PlusPlus::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Xoshiro256PlusPlus::new(0);
        let mut b = Xoshiro256PlusPlus::new(1);
        let same = (0..10).all(|_| a.next_u64() == b.next_u64());
        assert!(!same);
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Xoshiro256PlusPlus::new(123);
        let mut v: Vec<usize> = (0..100).collect();
        rng.shuffle(&mut v);
        v.sort();
        assert_eq!(v, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn shuffle_replay_matches_sequential() {
        let seed = 0xdeadbeef19260817u64;
        let pool = 20;

        // Generate the full sequence by running through 3 passes
        let mut rng1 = Xoshiro256PlusPlus::new(seed);
        let mut indices: Vec<usize> = (0..pool).collect();
        rng1.shuffle(&mut indices);
        let pass0 = indices.clone();
        rng1.shuffle(&mut indices);
        let pass1 = indices.clone();
        rng1.shuffle(&mut indices);
        let pass2 = indices.clone();

        // Now replay through 1 full pass and verify pass 1 matches
        let mut rng2 = Xoshiro256PlusPlus::new(seed);
        let mut indices2: Vec<usize> = (0..pool).collect();
        rng2.shuffle(&mut indices2); // initial
        assert_eq!(indices2, pass0);
        // Replay 1 reshuffle (as if resuming from sample `pool`)
        rng2.shuffle(&mut indices2);
        assert_eq!(indices2, pass1);
        // One more
        rng2.shuffle(&mut indices2);
        assert_eq!(indices2, pass2);
    }
}
