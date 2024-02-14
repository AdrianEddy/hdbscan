use std::cmp;
use crate::distance::DistanceMetric;

const MIN_CLUSTER_SIZE_DEFAULT: usize = 5;
const ALLOW_SINGLE_CLUSTER_DEFAULT: bool = false;
const EPSILON_DEFAULT: f64 = 0.0;
const DISTANCE_METRIC_DEFAULT: DistanceMetric = DistanceMetric::Euclidean;

/// A wrapper around the various hyper parameters used in HDBSCAN clustering.
/// Only use if you want to tune hyper parameters. Otherwise use `Hdbscan::default()` to 
/// instantiate the model with default hyper parameters.
pub struct HdbscanHyperParams {
    pub(crate) min_cluster_size: usize,
    pub(crate) allow_single_cluster: bool,
    pub(crate) min_samples: usize,
    pub(crate) epsilon: f64,
    pub(crate) dist_metric: DistanceMetric,
}

/// Builder object to set custom hyper parameters.
pub struct HyperParamBuilder {
    min_cluster_size: Option<usize>,
    allow_single_cluster: Option<bool>,
    min_samples: Option<usize>,
    epsilon: Option<f64>,
    dist_metric: Option<DistanceMetric>,
}

impl HdbscanHyperParams {
    pub(crate) fn default() -> Self {
        Self::builder().build()
    }

    /// Enters the builder pattern, allowing custom hyper parameters to be set using
    /// various setter methods.
    /// 
    /// # Returns 
    /// * the hyper parameter configuration builder
    pub fn builder() -> HyperParamBuilder {
        HyperParamBuilder {
            min_cluster_size: None,
            allow_single_cluster: None,
            min_samples: None,
            epsilon: None,
            dist_metric: None,
        }
    }
}

impl HyperParamBuilder {

    /// Sets the minimum cluster size - the minimum number of samples for a group of
    /// data points to be considered a cluster. If a grouping of data points has fewer
    /// members than this, then they will be considered noise.
    /// This should be considered the main hyper parameter for changing the results of clustering.
    /// Defaults to 5.
    ///
    /// # Parameters
    /// * min_cluster_size - the minimum cluster size
    ///
    /// # Returns
    /// * the hyper parameter configuration builder
    pub fn min_cluster_size(mut self, min_cluster_size: usize) -> HyperParamBuilder {
        self.min_cluster_size = Some(min_cluster_size);
        self
    }

    /// Sets whether to allow one single cluster (i.e. the root or top cluster). Only set
    /// this to true if you feel there being one cluster is correct for your dataset.
    /// Defaults to false.
    ///
    /// # Parameters
    /// * allow_single_cluster - whether to allow a single cluster
    ///
    /// # Returns
    /// * the hyper parameter configuration builder
    pub fn allow_single_cluster(mut self, allow_single_cluster: bool) -> HyperParamBuilder {
        self.allow_single_cluster = Some(allow_single_cluster);
        self
    }


    /// Sets min samples. HDBSCAN calculates the core distances between points as a first step
    /// in clustering. The core distance is the distance to the Kth neighbour using a nearest
    /// neighbours algorithm, where k = min_samples. Defaults to min_cluster_size.
    ///
    /// # Parameters
    /// * min_cluster_size - the number of neighbourhood points considered in distances
    ///
    /// # Returns
    /// * the hyper parameter configuration builder
    pub fn min_samples(mut self, min_samples: usize) -> HyperParamBuilder {
        self.min_samples = Some(min_samples);
        self
    }

    /// NOT IMPLEMENTED YET. WILL PANIC IF CALL.
    /// Sets cluster selection epsilon. Epsilon is a density threshold, below which point
    /// clusters will not be considered. By not setting this value, HDBSCAN will select the clusters
    /// that persist the longest at all densities (i.e. the longest lived clusters). Setting a 
    /// Defaults to 0.0, allowing clusters of all densities.
    ///
    /// # Parameters
    /// * epsilon - the minimum density, below which clusters will be discarded.
    ///
    /// # Returns
    /// * the hyper parameter configuration builder
    pub fn epsilon(mut self, epsilon: f64) -> HyperParamBuilder {
        todo!()
    }

    /// Sets the distance metric. HDBSCAN uses this metric to calculate the distance between data points.
    /// Defaults to Euclidean. Options are defined by the DistanceMetric enum.
    ///
    /// # Parameters
    /// * dist_metric - the distance metric
    ///
    /// # Returns
    /// * the hyper parameter configuration builder
    pub fn dist_metric(mut self, dist_metric: DistanceMetric) -> HyperParamBuilder {
        self.dist_metric = Some(dist_metric);
        self
    }

    /// Finishes the building of the hyper parameter configuration. A call to this method is required to 
    /// exist the builder pattern and complete the construction of the hyper parameters.
    ///
    /// # Returns
    /// * The completed HDBSCAN hyper parameter configuration.
    pub fn build(self) -> HdbscanHyperParams {
        let mut min_cluster_size = self.min_cluster_size.unwrap_or(MIN_CLUSTER_SIZE_DEFAULT);
        // Must be at least 2 data points to make a cluster
        min_cluster_size = cmp::max(min_cluster_size, 2);

        let mut min_samples = self.min_samples.unwrap_or(min_cluster_size);
        // Can't be less than 1
        min_samples = cmp::max(min_samples, 1);

        let mut epsilon = self.epsilon.unwrap_or(EPSILON_DEFAULT);
        // Can't be less than 0.0
        epsilon = epsilon.max(0.0);

        HdbscanHyperParams {
            min_cluster_size,
            allow_single_cluster: self.allow_single_cluster.unwrap_or(ALLOW_SINGLE_CLUSTER_DEFAULT),
            min_samples,
            epsilon,
            dist_metric: self.dist_metric.unwrap_or(DISTANCE_METRIC_DEFAULT),
        }
    }

}
