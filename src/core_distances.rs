use crate::{distance, DistanceMetric};
use num_traits::Float;
use rayon::prelude::*;

/// The nearest neighbour algorithm options
#[derive(Debug, Clone, PartialEq)]
pub enum NnAlgorithm {
    /// HDBSCAN internally selects the nearest neighbour based on size
    /// and dimensionality of the input data
    Auto,
    /// Computes a distance matrix between each point and all others
    BruteForce,
    /// K-dimensional tree algorithm.
    KdTree,
}

pub(crate) trait CoreDistance {
    fn calc_core_distances<T: Float + Send + Sync>(
        data: &[Vec<T>],
        k: usize,
        dist_metric: DistanceMetric,
    ) -> Vec<T>;
}

pub(crate) struct BruteForce;

impl CoreDistance for BruteForce {
    fn calc_core_distances<T: Float + Send + Sync>(
        data: &[Vec<T>],
        k: usize,
        dist_metric: DistanceMetric,
    ) -> Vec<T> {
        let dist_matrix = calc_pairwise_distances_parallel(data, distance::get_dist_func(&dist_metric));
        get_core_distances_from_matrix(&dist_matrix, k)
    }
}

fn calc_pairwise_distances_parallel<T, F>(data: &[Vec<T>], dist_func: F) -> Vec<Vec<T>>
where
    T: Float + Send + Sync,
    F: Fn(&[T], &[T]) -> T + Sync,
{
    let n_samples = data.len();

    (0..n_samples)
        .into_par_iter()
        .map(|i| {
            (0..n_samples)
                .into_par_iter()
                .map(|j| dist_func(&data[i], &data[j]))
                .collect()
        })
        .collect()
}

pub(crate) fn get_core_distances_from_matrix<T: Float + Send + Sync>(dist_matrix: &[Vec<T>], k: usize) -> Vec<T> {
    dist_matrix
        .par_iter()
        .map(|distances| {
            let mut sorted_distances = distances.clone();
            sorted_distances.par_sort_unstable_by(|a, b|
                a.partial_cmp(b).expect("Invalid float")
            );
            sorted_distances[k - 1]
        })
        .collect()
}

pub(crate) struct KdTree;

impl CoreDistance for KdTree {
    fn calc_core_distances<T: Float + Send + Sync>(
        data: &[Vec<T>],
        k: usize,
        dist_metric: DistanceMetric,
    ) -> Vec<T> {
        let mut tree: kdtree::KdTree<T, usize, &Vec<T>> = kdtree::KdTree::new(data[0].len());
        data.iter()
            .enumerate()
            .for_each(|(n, datapoint)| tree.add(datapoint, n).expect("Failed to add to KdTree"));

        let dist_func = distance::get_dist_func(&dist_metric);
        data.par_iter()
            .map(|datapoint| {
                let result = tree
                    .nearest(datapoint, k, &dist_func)
                    .expect("Failed to find neighbours");
                result
                    .into_iter()
                    .map(|(dist, _idx)| dist)
                    .last()
                    .expect("Failed to find neighbours")
            })
            .collect()
    }
}

impl BruteForce {
    /// Direct parallel implementation without building full distance matrix
    /// More memory efficient for large datasets
    pub fn calc_core_distances_direct<T: Float + Send + Sync>(
        data: &[Vec<T>],
        k: usize,
        dist_metric: DistanceMetric,
    ) -> Vec<T> {
        let dist_func = distance::get_dist_func(&dist_metric);

        data.par_iter()
            .map(|point| {
                let mut distances: Vec<T> = data
                    .par_iter()
                    .map(|other| dist_func(point, other))
                    .collect();

                distances.par_sort_unstable_by(|a, b|
                    a.partial_cmp(b).expect("Invalid float")
                );

                distances[k - 1]
            })
            .collect()
    }

    /// Chunked implementation to balance parallelism overhead
    pub fn calc_core_distances_chunked<T: Float + Send + Sync>(
        data: &[Vec<T>],
        k: usize,
        dist_metric: DistanceMetric,
        chunk_size: usize,
    ) -> Vec<T> {
        let dist_func = distance::get_dist_func(&dist_metric);

        data.par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.iter().map(|point| {
                    let mut distances: Vec<T> = data
                        .iter()
                        .map(|other| dist_func(point, other))
                        .collect();

                    distances.sort_unstable_by(|a, b|
                        a.partial_cmp(b).expect("Invalid float")
                    );

                    distances[k - 1]
                }).collect::<Vec<_>>()
            })
            .collect()
    }
}
