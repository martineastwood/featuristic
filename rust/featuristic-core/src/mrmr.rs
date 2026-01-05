//! Maximum Relevance Minimum Redundancy (mRMR) feature selection

use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// mRMR feature selector
pub struct MRMRSelector {
    k: usize,
}

impl MRMRSelector {
    /// Create a new mRMR selector
    pub fn new(num_features: usize) -> Self {
        Self { k: num_features }
    }

    /// Select features using mRMR
    pub fn select(
        &self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> Vec<usize> {
        let n_features = X.ncols();
        let k_to_select = self.k.min(n_features);

        // Compute relevance (correlation with target)
        let relevance = self.compute_relevance(X, y);

        // Compute redundancy matrix (correlation between features)
        let redundancy = self.compute_redundancy_matrix(X);

        // Greedy selection
        let mut selected = Vec::with_capacity(k_to_select);

        // First feature: max relevance
        let first = relevance
            .iter()
            .enumerate()
            .filter(|(_, &r)| r.is_finite())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        selected.push(first);

        // Incremental selection
        for _ in 1..k_to_select {
            let best = self.select_next_feature(&selected, &relevance, &redundancy);
            selected.push(best);
        }

        selected
    }

    /// Compute relevance (correlation with target) for all features
    fn compute_relevance(&self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Vec<f64> {
        (0..X.ncols())
            .into_par_iter()
            .map(|i| Self::correlation(&X.column(i), y))
            .collect()
    }

    /// Compute redundancy matrix (correlation between all feature pairs)
    /// Uses parallel computation
    fn compute_redundancy_matrix(&self, X: &ArrayView2<f64>) -> Array2<f64> {
        let n_features = X.ncols();
        let mut corr = Array2::zeros((n_features, n_features));

        // Compute correlations in parallel, collecting results
        let correlations: Vec<(usize, usize, f64)> = (0..n_features)
            .into_par_iter()
            .flat_map(|i| {
                (i + 1..n_features)
                    .into_par_iter()
                    .map(move |j| {
                        let col_i = X.column(i);
                        let col_j = X.column(j);
                        let corr_val = Self::correlation(&col_i, &col_j);
                        (i, j, corr_val)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Fill the matrix (serial, but fast)
        for (i, j, val) in correlations {
            corr[[i, j]] = val;
            corr[[j, i]] = val;
        }

        // Diagonal is 1.0 (self-correlation)
        for i in 0..n_features {
            corr[[i, i]] = 1.0;
        }

        corr
    }

    /// Select next feature using mRMR criterion: max(Relevance - AvgRedundancy)
    fn select_next_feature(
        &self,
        selected: &[usize],
        relevance: &[f64],
        redundancy: &Array2<f64>,
    ) -> usize {
        let n_features = relevance.len();

        // Find unselected feature with maximum mRMR score
        (0..n_features)
            .filter(|i| !selected.contains(i))
            .map(|candidate| {
                // Average redundancy with selected features
                let avg_redundancy: f64 = selected
                    .iter()
                    .map(|&s| redundancy[[candidate, s]].abs())
                    .sum::<f64>()
                    / selected.len() as f64;

                // mRMR score: relevance - redundancy
                let relevance_score = relevance[candidate];
                let mrmr_score = if relevance_score.is_finite() {
                    relevance_score - avg_redundancy
                } else {
                    f64::NEG_INFINITY
                };

                (candidate, mrmr_score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Compute Pearson correlation between two vectors
    #[inline]
    fn correlation(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let n = a.len();

        if n == 0 || b.len() == 0 {
            return 0.0;
        }

        let mean_a = a.mean().unwrap();
        let mean_b = b.mean().unwrap();

        // Compute correlation using vectorized operations
        let mut numerator = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;

        for i in 0..n {
            let diff_a = a[i] - mean_a;
            let diff_b = b[i] - mean_b;
            numerator += diff_a * diff_b;
            var_a += diff_a * diff_a;
            var_b += diff_b * diff_b;
        }

        let denominator = var_a.sqrt() * var_b.sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            (numerator / denominator).abs() // Return absolute correlation
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_correlation() {
        let a = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let corr = MRMRSelector::correlation(&a.view(), &b.view());
        assert!((corr - 1.0).abs() < 1e-10); // Perfect correlation

        let c = arr1(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        let corr_neg = MRMRSelector::correlation(&a.view(), &c.view());
        assert!((corr_neg - 1.0).abs() < 1e-10); // Absolute correlation
    }

    #[test]
    fn test_mrmr_selection() {
        // Create features with different correlations to target
        // Feature 0: highly correlated with target
        // Feature 1: moderately correlated with target
        // Feature 2: low correlation with target
        let x = arr2(&[
            [1.0, 5.0, 0.5],
            [2.0, 3.0, 1.5],
            [3.0, 7.0, 2.0],
            [4.0, 2.0, 1.0],
            [5.0, 8.0, 2.5],
        ]);
        let y = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let selector = MRMRSelector::new(2);
        let selected = selector.select(&x.view(), &y.view());

        assert_eq!(selected.len(), 2);
        // First feature should be selected (highest correlation with y)
        assert!(selected.contains(&0));
        // Should select 2 features
        assert!(selected.len() == 2);
    }

    #[test]
    fn test_redundancy_matrix() {
        let X = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]);

        let selector = MRMRSelector::new(2);
        let redundancy = selector.compute_redundancy_matrix(&X.view());

        assert_eq!(redundancy.shape(), &[3, 3]);
        // Diagonal should be 1.0
        for i in 0..3 {
            assert!((redundancy[[i, i]] - 1.0).abs() < 1e-10);
        }
    }
}
