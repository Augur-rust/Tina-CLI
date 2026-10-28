use std::fmt;
use std::path::Path;

use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::{ExactMetric, ExternalIndexIdentity, RetrievalError, RETRIEVAL_MAX_HITS};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UsearchIndexConfig {
    pub identity: ExternalIndexIdentity,
    pub metric: ExactMetric,
    pub capacity: usize,
    pub build_threads: usize,
    pub connectivity: usize,
    pub expansion_add: usize,
    pub expansion_search: usize,
}

impl UsearchIndexConfig {
    pub fn new(identity: ExternalIndexIdentity, metric: ExactMetric, capacity: usize) -> Self {
        Self {
            identity,
            metric,
            capacity,
            build_threads: 0,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
        }
    }

    fn validate(self) -> Result<(), UsearchError> {
        self.identity.validate().map_err(UsearchError::Contract)?;
        if self.capacity == 0 {
            return Err(UsearchError::ZeroCapacity);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UsearchAnnMatch {
    pub document_id: u64,
    pub distance: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UsearchError {
    Contract(RetrievalError),
    ZeroCapacity,
    InvalidK,
    Dimension { expected: usize, actual: usize },
    NonFiniteValue { dimension: usize },
    NonFiniteDistance { index: usize },
    ResultShape { keys: usize, distances: usize },
    MissingDocument { document_id: u64 },
    NonUtf8Path,
    Backend(String),
}

impl fmt::Display for UsearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "USEARCH adapter failure: {self:?}")
    }
}

impl std::error::Error for UsearchError {}

pub struct UsearchAnnIndex {
    identity: ExternalIndexIdentity,
    metric: ExactMetric,
    inner: Index,
}

impl UsearchAnnIndex {
    pub fn new(config: UsearchIndexConfig) -> Result<Self, UsearchError> {
        config.validate()?;
        let inner = new_inner(config)?;
        if config.build_threads == 0 {
            inner.reserve(config.capacity).map_err(backend_error)?;
        } else {
            inner
                .reserve_capacity_and_threads(config.capacity, config.build_threads)
                .map_err(backend_error)?;
        }
        Ok(Self {
            identity: config.identity,
            metric: config.metric,
            inner,
        })
    }

    pub fn load(config: UsearchIndexConfig, path: &Path) -> Result<Self, UsearchError> {
        config.validate()?;
        let inner = new_inner(config)?;
        inner.load(path_string(path)?).map_err(backend_error)?;
        if inner.dimensions() != config.identity.dimensions as usize {
            return Err(UsearchError::Dimension {
                expected: config.identity.dimensions as usize,
                actual: inner.dimensions(),
            });
        }
        if inner.capacity() < config.capacity {
            if config.build_threads == 0 {
                inner.reserve(config.capacity).map_err(backend_error)?;
            } else {
                inner
                    .reserve_capacity_and_threads(config.capacity, config.build_threads)
                    .map_err(backend_error)?;
            }
        }
        Ok(Self {
            identity: config.identity,
            metric: config.metric,
            inner,
        })
    }

    pub fn identity(&self) -> ExternalIndexIdentity {
        self.identity
    }

    pub fn metric(&self) -> ExactMetric {
        self.metric
    }

    pub fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    pub fn len(&self) -> usize {
        self.inner.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add(&self, document_id: u64, embedding: &[f32]) -> Result<(), UsearchError> {
        validate_embedding(embedding, self.dimensions())?;
        self.inner
            .add(document_id, embedding)
            .map_err(backend_error)
    }

    pub fn save(&self, path: &Path) -> Result<(), UsearchError> {
        self.inner.save(path_string(path)?).map_err(backend_error)
    }

    pub fn remove(&self, document_id: u64) -> Result<bool, UsearchError> {
        self.inner
            .remove(document_id)
            .map(|removed| removed != 0)
            .map_err(backend_error)
    }

    pub fn get(&self, document_id: u64) -> Result<Vec<f32>, UsearchError> {
        let mut embedding = vec![0.0_f32; self.dimensions()];
        let found = self
            .inner
            .get(document_id, &mut embedding)
            .map_err(backend_error)?;
        if found == 0 {
            return Err(UsearchError::MissingDocument { document_id });
        }
        if found != 1 {
            return Err(UsearchError::Backend(format!(
                "non-multi index returned {found} vectors for key {document_id}"
            )));
        }
        Ok(embedding)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<UsearchAnnMatch>, UsearchError> {
        if k == 0 || k > RETRIEVAL_MAX_HITS {
            return Err(UsearchError::InvalidK);
        }
        validate_embedding(query, self.dimensions())?;
        let matches = self.inner.search(query, k).map_err(backend_error)?;
        if matches.keys.len() != matches.distances.len() {
            return Err(UsearchError::ResultShape {
                keys: matches.keys.len(),
                distances: matches.distances.len(),
            });
        }
        let mut output = Vec::with_capacity(matches.keys.len());
        for (index, (document_id, distance)) in
            matches.keys.into_iter().zip(matches.distances).enumerate()
        {
            if !distance.is_finite() {
                return Err(UsearchError::NonFiniteDistance { index });
            }
            output.push(UsearchAnnMatch {
                document_id,
                distance: normalize_zero(distance),
            });
        }
        output.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.document_id.cmp(&right.document_id))
        });
        Ok(output)
    }
}

fn new_inner(config: UsearchIndexConfig) -> Result<Index, UsearchError> {
    let options = IndexOptions {
        dimensions: config.identity.dimensions as usize,
        metric: usearch_metric(config.metric),
        quantization: ScalarKind::F32,
        connectivity: config.connectivity,
        expansion_add: config.expansion_add,
        expansion_search: config.expansion_search,
        multi: false,
    };
    Index::new(&options).map_err(backend_error)
}

fn usearch_metric(metric: ExactMetric) -> MetricKind {
    match metric {
        ExactMetric::Cosine => MetricKind::Cos,
        ExactMetric::InnerProduct => MetricKind::IP,
        ExactMetric::SquaredL2 => MetricKind::L2sq,
    }
}

fn validate_embedding(embedding: &[f32], expected: usize) -> Result<(), UsearchError> {
    if embedding.len() != expected {
        return Err(UsearchError::Dimension {
            expected,
            actual: embedding.len(),
        });
    }
    if let Some(dimension) = embedding.iter().position(|value| !value.is_finite()) {
        return Err(UsearchError::NonFiniteValue { dimension });
    }
    Ok(())
}

fn normalize_zero(value: f32) -> f32 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn backend_error(error: impl fmt::Display) -> UsearchError {
    UsearchError::Backend(error.to_string())
}

fn path_string(path: &Path) -> Result<&str, UsearchError> {
    path.to_str().ok_or(UsearchError::NonUtf8Path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(dimensions: u32) -> ExternalIndexIdentity {
        ExternalIndexIdentity {
            dimensions,
            revision_sha256: [1; 32],
            license_sha256: [2; 32],
        }
    }

    #[test]
    fn invalid_identity_fails_before_native_construction() {
        let error = UsearchAnnIndex::new(UsearchIndexConfig::new(
            ExternalIndexIdentity {
                dimensions: 2,
                revision_sha256: [0; 32],
                license_sha256: [2; 32],
            },
            ExactMetric::Cosine,
            4,
        ));
        assert!(matches!(
            error,
            Err(UsearchError::Contract(
                RetrievalError::ExternalIdentityMismatch
            ))
        ));
    }

    #[test]
    fn f32_index_returns_bounded_candidates() {
        let index =
            UsearchAnnIndex::new(UsearchIndexConfig::new(identity(2), ExactMetric::Cosine, 4))
                .unwrap();
        index.add(30, &[0.0, 1.0]).unwrap();
        index.add(10, &[1.0, 0.0]).unwrap();
        index.add(20, &[0.8, 0.2]).unwrap();
        let matches = index.search(&[1.0, 0.0], 2).unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].document_id, 10);
        assert_eq!(index.identity(), identity(2));
        assert_eq!(index.metric(), ExactMetric::Cosine);
    }

    #[test]
    fn malformed_vectors_and_k_fail_at_rust_boundary() {
        let index = UsearchAnnIndex::new(UsearchIndexConfig::new(
            identity(2),
            ExactMetric::SquaredL2,
            2,
        ))
        .unwrap();
        assert!(matches!(
            index.add(1, &[1.0]),
            Err(UsearchError::Dimension { .. })
        ));
        assert!(matches!(
            index.search(&[f32::NAN, 0.0], 1),
            Err(UsearchError::NonFiniteValue { .. })
        ));
        assert_eq!(index.search(&[0.0, 0.0], 0), Err(UsearchError::InvalidK));
    }

    #[test]
    fn saved_index_round_trips_vectors_and_search() {
        let path = std::env::temp_dir().join(format!(
            "cq-usearch-{}-{}.usearch",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = UsearchIndexConfig::new(identity(2), ExactMetric::Cosine, 4);
        let index = UsearchAnnIndex::new(config).unwrap();
        index.add(10, &[1.0, 0.0]).unwrap();
        index.add(20, &[0.0, 1.0]).unwrap();
        index.save(&path).unwrap();

        let loaded = UsearchAnnIndex::load(config, &path).unwrap();
        assert_eq!(loaded.get(10).unwrap(), vec![1.0, 0.0]);
        assert_eq!(loaded.search(&[1.0, 0.0], 1).unwrap()[0].document_id, 10);
        assert!(loaded.remove(20).unwrap());
        assert!(!loaded.remove(20).unwrap());
        std::fs::remove_file(path).unwrap();
    }
}
