use std::fmt;

pub const RETRIEVAL_MAX_HITS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetrievalBackend {
    Disabled,
    Fixture,
    ExternalUsearch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetrievalError {
    ZeroCapacity,
    InvalidScoreThreshold,
    Disabled,
    Capacity {
        required: usize,
        capacity: usize,
    },
    NonFiniteScore {
        index: usize,
    },
    FuturePosition {
        index: usize,
        position: u32,
        current: u32,
    },
    DuplicateDocument {
        index: usize,
    },
    DuplicatePosition {
        index: usize,
    },
    ScoreOrder {
        index: usize,
    },
    InvalidGold {
        index: usize,
    },
    InvalidK,
    ExternalIdentityMismatch,
}

impl fmt::Display for RetrievalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid retrieval contract: {self:?}")
    }
}

impl std::error::Error for RetrievalError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetrievalHit {
    pub document_id: u64,
    pub position: u32,
    pub score: f32,
}

impl Default for RetrievalHit {
    fn default() -> Self {
        Self {
            document_id: 0,
            position: 0,
            score: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetrievalConfig {
    pub backend: RetrievalBackend,
    pub max_hits: usize,
    min_score_bits: u32,
}

impl RetrievalConfig {
    pub fn new(
        backend: RetrievalBackend,
        max_hits: usize,
        min_score: f32,
    ) -> Result<Self, RetrievalError> {
        if max_hits == 0 || max_hits > RETRIEVAL_MAX_HITS {
            return Err(RetrievalError::Capacity {
                required: max_hits,
                capacity: RETRIEVAL_MAX_HITS,
            });
        }
        if !min_score.is_finite() {
            return Err(RetrievalError::InvalidScoreThreshold);
        }
        Ok(Self {
            backend,
            max_hits,
            min_score_bits: min_score.to_bits(),
        })
    }

    pub fn min_score(self) -> f32 {
        f32::from_bits(self.min_score_bits)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExternalIndexIdentity {
    pub dimensions: u32,
    pub revision_sha256: [u8; 32],
    pub license_sha256: [u8; 32],
}

impl ExternalIndexIdentity {
    pub fn validate(self) -> Result<(), RetrievalError> {
        if self.dimensions == 0 || self.revision_sha256 == [0; 32] || self.license_sha256 == [0; 32]
        {
            return Err(RetrievalError::ExternalIdentityMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CausalHitSet<const N: usize> {
    hits: [RetrievalHit; N],
    len: usize,
}

impl<const N: usize> Default for CausalHitSet<N> {
    fn default() -> Self {
        Self {
            hits: [RetrievalHit::default(); N],
            len: 0,
        }
    }
}

impl<const N: usize> CausalHitSet<N> {
    pub fn admit(
        &mut self,
        backend: RetrievalBackend,
        current_position: u32,
        min_score: f32,
        hits: &[RetrievalHit],
    ) -> Result<usize, RetrievalError> {
        if backend == RetrievalBackend::Disabled {
            return Err(RetrievalError::Disabled);
        }
        if !min_score.is_finite() {
            return Err(RetrievalError::InvalidScoreThreshold);
        }
        if hits.len() > N {
            return Err(RetrievalError::Capacity {
                required: hits.len(),
                capacity: N,
            });
        }
        for (index, hit) in hits.iter().enumerate() {
            if !hit.score.is_finite() {
                return Err(RetrievalError::NonFiniteScore { index });
            }
            if hit.position >= current_position {
                return Err(RetrievalError::FuturePosition {
                    index,
                    position: hit.position,
                    current: current_position,
                });
            }
            if hit.score < min_score || (index > 0 && hit.score > hits[index - 1].score) {
                return Err(RetrievalError::ScoreOrder { index });
            }
            if hits[..index]
                .iter()
                .any(|previous| previous.document_id == hit.document_id)
            {
                return Err(RetrievalError::DuplicateDocument { index });
            }
            if hits[..index]
                .iter()
                .any(|previous| previous.position == hit.position)
            {
                return Err(RetrievalError::DuplicatePosition { index });
            }
        }
        self.hits[..hits.len()].copy_from_slice(hits);
        self.hits[hits.len()..].fill(RetrievalHit::default());
        self.len = hits.len();
        Ok(self.len)
    }

    pub fn hits(&self) -> &[RetrievalHit] {
        &self.hits[..self.len]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparsePositionSet<const N: usize> {
    positions: [u32; N],
    len: usize,
}

impl<const N: usize> Default for SparsePositionSet<N> {
    fn default() -> Self {
        Self {
            positions: [0; N],
            len: 0,
        }
    }
}

impl<const N: usize> SparsePositionSet<N> {
    pub fn build(
        &mut self,
        current_position: u32,
        local_window: u32,
        retrieved: &CausalHitSet<N>,
    ) -> Result<usize, RetrievalError> {
        let start = current_position.saturating_sub(local_window);
        let mut next = [0u32; N];
        let mut next_len = 0;
        for position in start..current_position {
            if next_len == N {
                return Err(RetrievalError::Capacity {
                    required: next_len + 1,
                    capacity: N,
                });
            }
            next[next_len] = position;
            next_len += 1;
        }
        for hit in retrieved.hits() {
            if !next[..next_len].contains(&hit.position) {
                if next_len == N {
                    return Err(RetrievalError::Capacity {
                        required: next_len + 1,
                        capacity: N,
                    });
                }
                next[next_len] = hit.position;
                next_len += 1;
            }
        }
        next[..next_len].sort_unstable();
        self.positions = next;
        self.len = next_len;
        Ok(next_len)
    }

    pub fn positions(&self) -> &[u32] {
        &self.positions[..self.len]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecallAtK {
    pub matched: usize,
    pub relevant: usize,
    pub evaluated: usize,
    pub recall: f32,
}

pub fn recall_at_k(
    gold_document_ids: &[u64],
    hits: &[RetrievalHit],
    k: usize,
) -> Result<RecallAtK, RetrievalError> {
    if k == 0 || gold_document_ids.is_empty() {
        return Err(RetrievalError::InvalidK);
    }
    for (index, id) in gold_document_ids.iter().enumerate() {
        if *id == 0 || gold_document_ids[..index].contains(id) {
            return Err(RetrievalError::InvalidGold { index });
        }
    }
    let evaluated = k.min(hits.len());
    let matched = hits[..evaluated]
        .iter()
        .filter(|hit| gold_document_ids.contains(&hit.document_id))
        .count();
    Ok(RecallAtK {
        matched,
        relevant: gold_document_ids.len(),
        evaluated,
        recall: matched as f32 / gold_document_ids.len() as f32,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenefitDisposition {
    Qualified,
    NoGoRecall,
    NoGoQuality,
    NoGoLatency,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenefitEvidence {
    pub baseline_quality: f32,
    pub retrieval_quality: f32,
    pub baseline_wall_nanoseconds: u64,
    pub retrieval_wall_nanoseconds: u64,
}

impl BenefitEvidence {
    pub fn qualify(self, recall: RecallAtK, minimum_recall: f32) -> BenefitDisposition {
        if !minimum_recall.is_finite()
            || !self.baseline_quality.is_finite()
            || !self.retrieval_quality.is_finite()
            || recall.recall < minimum_recall
        {
            return BenefitDisposition::NoGoRecall;
        }
        if self.retrieval_quality < self.baseline_quality {
            return BenefitDisposition::NoGoQuality;
        }
        if self.retrieval_wall_nanoseconds >= self.baseline_wall_nanoseconds {
            return BenefitDisposition::NoGoLatency;
        }
        BenefitDisposition::Qualified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hits() -> [RetrievalHit; 3] {
        [
            RetrievalHit {
                document_id: 10,
                position: 3,
                score: 0.9,
            },
            RetrievalHit {
                document_id: 11,
                position: 7,
                score: 0.8,
            },
            RetrievalHit {
                document_id: 12,
                position: 1,
                score: 0.7,
            },
        ]
    }

    #[test]
    fn disabled_or_future_hits_fail_before_mutation() {
        let mut set = CausalHitSet::<8>::default();
        assert_eq!(
            set.admit(RetrievalBackend::Disabled, 10, 0.0, &hits()),
            Err(RetrievalError::Disabled)
        );
        let future = [RetrievalHit {
            position: 10,
            ..hits()[0]
        }];
        assert!(set
            .admit(RetrievalBackend::Fixture, 10, 0.0, &future)
            .is_err());
        assert!(set.hits().is_empty());
    }

    #[test]
    fn causal_set_merges_local_and_retrieved_in_position_order() {
        let mut hits_set = CausalHitSet::<8>::default();
        hits_set
            .admit(RetrievalBackend::Fixture, 10, 0.0, &hits())
            .unwrap();
        let mut positions = SparsePositionSet::<8>::default();
        positions.build(10, 2, &hits_set).unwrap();
        assert_eq!(positions.positions(), &[1, 3, 7, 8, 9]);
    }

    #[test]
    fn recall_and_benefit_are_separate_gates() {
        let hits = hits();
        let recall = recall_at_k(&[10, 12, 99], &hits, 3).unwrap();
        assert_eq!(recall.matched, 2);
        let evidence = BenefitEvidence {
            baseline_quality: 0.9,
            retrieval_quality: 0.9,
            baseline_wall_nanoseconds: 100,
            retrieval_wall_nanoseconds: 90,
        };
        assert_eq!(evidence.qualify(recall, 0.5), BenefitDisposition::Qualified);
        assert_eq!(
            evidence.qualify(recall, 0.9),
            BenefitDisposition::NoGoRecall
        );
    }

    #[test]
    fn score_order_and_duplicates_fail_closed() {
        let mut set = CausalHitSet::<8>::default();
        let mut invalid = hits();
        invalid[1].score = 0.95;
        assert_eq!(
            set.admit(RetrievalBackend::Fixture, 10, 0.0, &invalid),
            Err(RetrievalError::ScoreOrder { index: 1 })
        );
        invalid = hits();
        invalid[1].document_id = invalid[0].document_id;
        assert_eq!(
            set.admit(RetrievalBackend::Fixture, 10, 0.0, &invalid),
            Err(RetrievalError::DuplicateDocument { index: 1 })
        );
    }

    #[test]
    fn identity_and_config_are_bounded() {
        assert_eq!(
            ExternalIndexIdentity {
                dimensions: 128,
                revision_sha256: [0; 32],
                license_sha256: [1; 32],
            }
            .validate(),
            Err(RetrievalError::ExternalIdentityMismatch)
        );
        assert!(RetrievalConfig::new(
            RetrievalBackend::ExternalUsearch,
            RETRIEVAL_MAX_HITS + 1,
            0.0,
        )
        .is_err());
    }
}
