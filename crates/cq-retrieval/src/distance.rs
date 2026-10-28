use std::cmp::Ordering;
use std::fmt;

use crate::RetrievalHit;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactMetric {
    Cosine,
    InnerProduct,
    SquaredL2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceKernel {
    Scalar,
    Avx2Fma,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorRole {
    Query,
    Candidate { index: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceError {
    EmptyVector,
    InvalidLimit,
    Capacity {
        required: usize,
        capacity: usize,
    },
    Dimension {
        role: VectorRole,
        expected: usize,
        actual: usize,
    },
    NonFiniteValue {
        role: VectorRole,
        dimension: usize,
    },
    ZeroNorm {
        role: VectorRole,
    },
    NonFiniteScore {
        candidate: usize,
    },
    DuplicateDocument {
        index: usize,
    },
    DuplicatePosition {
        index: usize,
    },
}

impl fmt::Display for DistanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid exact-distance contract: {self:?}")
    }
}

impl std::error::Error for DistanceError {}

#[derive(Clone, Copy, Debug)]
pub struct ExactCandidate<'embedding> {
    pub document_id: u64,
    pub position: u32,
    pub embedding: &'embedding [f32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactRerank<const N: usize> {
    hits: [RetrievalHit; N],
    len: usize,
}

impl<const N: usize> Default for ExactRerank<N> {
    fn default() -> Self {
        Self {
            hits: [RetrievalHit::default(); N],
            len: 0,
        }
    }
}

impl<const N: usize> ExactRerank<N> {
    pub fn rerank(
        &mut self,
        metric: ExactMetric,
        query: &[f32],
        candidates: &[ExactCandidate<'_>],
        limit: usize,
    ) -> Result<usize, DistanceError> {
        if limit == 0 {
            return Err(DistanceError::InvalidLimit);
        }
        let required = limit.min(candidates.len());
        if required > N {
            return Err(DistanceError::Capacity {
                required,
                capacity: N,
            });
        }
        validate_vector(query, VectorRole::Query, query.len())?;
        validate_candidate_set(metric, query, candidates)?;

        let mut next = [RetrievalHit::default(); N];
        let mut next_len = 0usize;
        for (index, candidate) in candidates.iter().enumerate() {
            let score =
                exact_score(metric, query, candidate.embedding).map_err(|error| match error {
                    DistanceError::ZeroNorm {
                        role: VectorRole::Candidate { .. },
                    } => DistanceError::ZeroNorm {
                        role: VectorRole::Candidate { index },
                    },
                    DistanceError::NonFiniteScore { .. } => {
                        DistanceError::NonFiniteScore { candidate: index }
                    }
                    other => other,
                })?;
            if !score.is_finite() {
                return Err(DistanceError::NonFiniteScore { candidate: index });
            }
            let hit = RetrievalHit {
                document_id: candidate.document_id,
                position: candidate.position,
                score: normalize_zero(score),
            };
            insert_ranked(&mut next, &mut next_len, required, hit);
        }

        self.hits = next;
        self.len = next_len;
        Ok(next_len)
    }

    pub fn hits(&self) -> &[RetrievalHit] {
        &self.hits[..self.len]
    }
}

pub fn selected_distance_kernel() -> DistanceKernel {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return DistanceKernel::Avx2Fma;
    }
    DistanceKernel::Scalar
}

pub fn exact_score(metric: ExactMetric, lhs: &[f32], rhs: &[f32]) -> Result<f32, DistanceError> {
    validate_pair(lhs, rhs)?;
    let score = match selected_distance_kernel() {
        DistanceKernel::Scalar => exact_score_scalar_unchecked(metric, lhs, rhs),
        DistanceKernel::Avx2Fma => {
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: selected_distance_kernel returns this branch only after
                // runtime AVX2 and FMA detection; validate_pair proved equal lengths.
                unsafe { exact_score_avx2_fma_unchecked(metric, lhs, rhs) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                exact_score_scalar_unchecked(metric, lhs, rhs)
            }
        }
    };
    finish_score(
        metric,
        score,
        VectorRole::Query,
        VectorRole::Candidate { index: 0 },
    )
}

pub fn exact_score_scalar(
    metric: ExactMetric,
    lhs: &[f32],
    rhs: &[f32],
) -> Result<f32, DistanceError> {
    validate_pair(lhs, rhs)?;
    finish_score(
        metric,
        exact_score_scalar_unchecked(metric, lhs, rhs),
        VectorRole::Query,
        VectorRole::Candidate { index: 0 },
    )
}

fn validate_pair(lhs: &[f32], rhs: &[f32]) -> Result<(), DistanceError> {
    validate_vector(lhs, VectorRole::Query, lhs.len())?;
    validate_vector(rhs, VectorRole::Candidate { index: 0 }, lhs.len())
}

fn validate_candidate_set(
    metric: ExactMetric,
    query: &[f32],
    candidates: &[ExactCandidate<'_>],
) -> Result<(), DistanceError> {
    if metric == ExactMetric::Cosine && squared_norm_scalar(query) == 0.0 {
        return Err(DistanceError::ZeroNorm {
            role: VectorRole::Query,
        });
    }
    for (index, candidate) in candidates.iter().enumerate() {
        let role = VectorRole::Candidate { index };
        validate_vector(candidate.embedding, role, query.len())?;
        if metric == ExactMetric::Cosine && squared_norm_scalar(candidate.embedding) == 0.0 {
            return Err(DistanceError::ZeroNorm { role });
        }
        if candidates[..index]
            .iter()
            .any(|previous| previous.document_id == candidate.document_id)
        {
            return Err(DistanceError::DuplicateDocument { index });
        }
        if candidates[..index]
            .iter()
            .any(|previous| previous.position == candidate.position)
        {
            return Err(DistanceError::DuplicatePosition { index });
        }
    }
    Ok(())
}

fn validate_vector(vector: &[f32], role: VectorRole, expected: usize) -> Result<(), DistanceError> {
    if expected == 0 {
        return Err(DistanceError::EmptyVector);
    }
    if vector.len() != expected {
        return Err(DistanceError::Dimension {
            role,
            expected,
            actual: vector.len(),
        });
    }
    if let Some(dimension) = vector.iter().position(|value| !value.is_finite()) {
        return Err(DistanceError::NonFiniteValue { role, dimension });
    }
    Ok(())
}

fn exact_score_scalar_unchecked(metric: ExactMetric, lhs: &[f32], rhs: &[f32]) -> RawScore {
    match metric {
        ExactMetric::InnerProduct => {
            let mut dot = 0.0f32;
            for (&left, &right) in lhs.iter().zip(rhs) {
                dot += left * right;
            }
            RawScore::InnerProduct(dot)
        }
        ExactMetric::SquaredL2 => {
            let mut squared_l2 = 0.0f32;
            for (&left, &right) in lhs.iter().zip(rhs) {
                let delta = left - right;
                squared_l2 += delta * delta;
            }
            RawScore::SquaredL2(squared_l2)
        }
        ExactMetric::Cosine => {
            let mut dot = 0.0f32;
            let mut lhs_squared = 0.0f32;
            let mut rhs_squared = 0.0f32;
            for (&left, &right) in lhs.iter().zip(rhs) {
                dot += left * right;
                lhs_squared += left * left;
                rhs_squared += right * right;
            }
            RawScore::Cosine {
                dot,
                lhs_squared,
                rhs_squared,
            }
        }
    }
}

fn squared_norm_scalar(vector: &[f32]) -> f32 {
    vector.iter().fold(0.0, |sum, value| sum + value * value)
}

#[derive(Clone, Copy, Debug)]
enum RawScore {
    InnerProduct(f32),
    SquaredL2(f32),
    Cosine {
        dot: f32,
        lhs_squared: f32,
        rhs_squared: f32,
    },
}

fn finish_score(
    metric: ExactMetric,
    raw: RawScore,
    lhs_role: VectorRole,
    rhs_role: VectorRole,
) -> Result<f32, DistanceError> {
    let score = match (metric, raw) {
        (ExactMetric::InnerProduct, RawScore::InnerProduct(dot)) => dot,
        (ExactMetric::SquaredL2, RawScore::SquaredL2(distance)) => -distance,
        (
            ExactMetric::Cosine,
            RawScore::Cosine {
                dot,
                lhs_squared,
                rhs_squared,
            },
        ) => {
            if lhs_squared == 0.0 {
                return Err(DistanceError::ZeroNorm { role: lhs_role });
            }
            if rhs_squared == 0.0 {
                return Err(DistanceError::ZeroNorm { role: rhs_role });
            }
            dot / (lhs_squared.sqrt() * rhs_squared.sqrt())
        }
        _ => unreachable!("metric and raw score are constructed together"),
    };
    if score.is_finite() {
        Ok(normalize_zero(score))
    } else {
        Err(DistanceError::NonFiniteScore { candidate: 0 })
    }
}

fn normalize_zero(value: f32) -> f32 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn insert_ranked<const N: usize>(
    output: &mut [RetrievalHit; N],
    len: &mut usize,
    limit: usize,
    hit: RetrievalHit,
) {
    if limit == 0 {
        return;
    }
    let insertion = output[..*len]
        .iter()
        .position(|current| ranks_before(&hit, current))
        .unwrap_or(*len);
    if insertion >= limit {
        return;
    }
    let new_len = (*len + 1).min(limit);
    for index in (insertion + 1..new_len).rev() {
        output[index] = output[index - 1];
    }
    output[insertion] = hit;
    *len = new_len;
}

fn ranks_before(lhs: &RetrievalHit, rhs: &RetrievalHit) -> bool {
    match lhs.score.total_cmp(&rhs.score) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => (lhs.document_id, lhs.position) < (rhs.document_id, rhs.position),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exact_score_avx2_fma_unchecked(
    metric: ExactMetric,
    lhs: &[f32],
    rhs: &[f32],
) -> RawScore {
    use std::arch::x86_64::*;

    let mut offset = 0usize;
    match metric {
        ExactMetric::InnerProduct => {
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            while offset + 16 <= lhs.len() {
                // SAFETY: The loop guard covers 16 lhs elements and validate_pair
                // established the identical rhs length; this function enables AVX2.
                let (lhs0, lhs1, rhs0, rhs1) = unsafe { load_two(lhs, rhs, offset) };
                sum0 = _mm256_fmadd_ps(lhs0, rhs0, sum0);
                sum1 = _mm256_fmadd_ps(lhs1, rhs1, sum1);
                offset += 16;
            }
            // SAFETY: This function's AVX2 target feature satisfies horizontal_sum.
            let mut dot = unsafe { horizontal_sum(_mm256_add_ps(sum0, sum1)) };
            for index in offset..lhs.len() {
                dot += lhs[index] * rhs[index];
            }
            RawScore::InnerProduct(dot)
        }
        ExactMetric::SquaredL2 => {
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            while offset + 16 <= lhs.len() {
                // SAFETY: The loop guard and equal-length validation cover both
                // 8-lane loads from each slice; AVX2 is enabled for this function.
                let (lhs0, lhs1, rhs0, rhs1) = unsafe { load_two(lhs, rhs, offset) };
                let delta0 = _mm256_sub_ps(lhs0, rhs0);
                let delta1 = _mm256_sub_ps(lhs1, rhs1);
                sum0 = _mm256_fmadd_ps(delta0, delta0, sum0);
                sum1 = _mm256_fmadd_ps(delta1, delta1, sum1);
                offset += 16;
            }
            // SAFETY: This function's AVX2 target feature satisfies horizontal_sum.
            let mut squared_l2 = unsafe { horizontal_sum(_mm256_add_ps(sum0, sum1)) };
            for index in offset..lhs.len() {
                let delta = lhs[index] - rhs[index];
                squared_l2 += delta * delta;
            }
            RawScore::SquaredL2(squared_l2)
        }
        ExactMetric::Cosine => {
            let mut dot0 = _mm256_setzero_ps();
            let mut dot1 = _mm256_setzero_ps();
            let mut lhs_squared0 = _mm256_setzero_ps();
            let mut lhs_squared1 = _mm256_setzero_ps();
            let mut rhs_squared0 = _mm256_setzero_ps();
            let mut rhs_squared1 = _mm256_setzero_ps();
            while offset + 16 <= lhs.len() {
                // SAFETY: The loop guard and equal-length validation cover all
                // four unaligned 8-lane loads; AVX2 is enabled here.
                let (lhs0, lhs1, rhs0, rhs1) = unsafe { load_two(lhs, rhs, offset) };
                dot0 = _mm256_fmadd_ps(lhs0, rhs0, dot0);
                dot1 = _mm256_fmadd_ps(lhs1, rhs1, dot1);
                lhs_squared0 = _mm256_fmadd_ps(lhs0, lhs0, lhs_squared0);
                lhs_squared1 = _mm256_fmadd_ps(lhs1, lhs1, lhs_squared1);
                rhs_squared0 = _mm256_fmadd_ps(rhs0, rhs0, rhs_squared0);
                rhs_squared1 = _mm256_fmadd_ps(rhs1, rhs1, rhs_squared1);
                offset += 16;
            }
            // SAFETY: This function's AVX2 target feature satisfies horizontal_sum.
            let mut dot = unsafe { horizontal_sum(_mm256_add_ps(dot0, dot1)) };
            // SAFETY: The accumulated vector is initialized and AVX2 is enabled.
            let mut lhs_squared =
                unsafe { horizontal_sum(_mm256_add_ps(lhs_squared0, lhs_squared1)) };
            // SAFETY: The accumulated vector is initialized and AVX2 is enabled.
            let mut rhs_squared =
                unsafe { horizontal_sum(_mm256_add_ps(rhs_squared0, rhs_squared1)) };
            for index in offset..lhs.len() {
                dot += lhs[index] * rhs[index];
                lhs_squared += lhs[index] * lhs[index];
                rhs_squared += rhs[index] * rhs[index];
            }
            RawScore::Cosine {
                dot,
                lhs_squared,
                rhs_squared,
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn load_two(
    lhs: &[f32],
    rhs: &[f32],
    offset: usize,
) -> (
    std::arch::x86_64::__m256,
    std::arch::x86_64::__m256,
    std::arch::x86_64::__m256,
    std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;

    // SAFETY: The caller guarantees both slices contain `offset + 16` elements.
    // Pointer additions therefore remain in-bounds, and loadu requires no alignment.
    unsafe {
        (
            _mm256_loadu_ps(lhs.as_ptr().add(offset)),
            _mm256_loadu_ps(lhs.as_ptr().add(offset + 8)),
            _mm256_loadu_ps(rhs.as_ptr().add(offset)),
            _mm256_loadu_ps(rhs.as_ptr().add(offset + 8)),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum(vector: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let mut lanes = [0.0f32; 8];
    // SAFETY: `lanes` provides eight contiguous writable f32 slots and this
    // function's target feature satisfies the AVX2 store intrinsic.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), vector) };
    lanes.into_iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_metrics_use_higher_is_better_scores() {
        let lhs = [1.0, 2.0, 3.0];
        let rhs = [1.0, 0.0, 3.0];
        assert_eq!(
            exact_score_scalar(ExactMetric::InnerProduct, &lhs, &rhs),
            Ok(10.0)
        );
        assert_eq!(
            exact_score_scalar(ExactMetric::SquaredL2, &lhs, &rhs),
            Ok(-4.0)
        );
        let cosine = exact_score_scalar(ExactMetric::Cosine, &lhs, &rhs).unwrap();
        assert!((cosine - 10.0 / (14.0f32 * 10.0).sqrt()).abs() < 1.0e-6);
    }

    #[test]
    fn invalid_vectors_fail_closed() {
        assert_eq!(
            exact_score(ExactMetric::InnerProduct, &[], &[]),
            Err(DistanceError::EmptyVector)
        );
        assert!(matches!(
            exact_score(ExactMetric::InnerProduct, &[1.0], &[1.0, 2.0]),
            Err(DistanceError::Dimension { .. })
        ));
        assert!(matches!(
            exact_score(ExactMetric::InnerProduct, &[f32::NAN], &[1.0]),
            Err(DistanceError::NonFiniteValue { .. })
        ));
        assert_eq!(
            exact_score(ExactMetric::Cosine, &[0.0, 0.0], &[1.0, 0.0]),
            Err(DistanceError::ZeroNorm {
                role: VectorRole::Query
            })
        );
    }

    #[test]
    fn runtime_dispatch_matches_scalar_with_tail() {
        let lhs: Vec<f32> = (0..37).map(|index| index as f32 * 0.03125 - 0.5).collect();
        let rhs: Vec<f32> = (0..37)
            .map(|index| 0.75 - index as f32 * 0.015625)
            .collect();
        for metric in [
            ExactMetric::Cosine,
            ExactMetric::InnerProduct,
            ExactMetric::SquaredL2,
        ] {
            let scalar = exact_score_scalar(metric, &lhs, &rhs).unwrap();
            let dispatched = exact_score(metric, &lhs, &rhs).unwrap();
            let tolerance = 2.0e-5 * scalar.abs().max(1.0);
            assert!((scalar - dispatched).abs() <= tolerance);
        }
    }

    #[test]
    fn rerank_is_exact_bounded_and_deterministic() {
        let query = [1.0, 0.0];
        let candidates = [
            ExactCandidate {
                document_id: 30,
                position: 3,
                embedding: &[0.0, 1.0],
            },
            ExactCandidate {
                document_id: 20,
                position: 2,
                embedding: &[1.0, 0.0],
            },
            ExactCandidate {
                document_id: 10,
                position: 1,
                embedding: &[1.0, 0.0],
            },
        ];
        let mut rerank = ExactRerank::<2>::default();
        assert_eq!(
            rerank.rerank(ExactMetric::Cosine, &query, &candidates, 2),
            Ok(2)
        );
        assert_eq!(
            rerank
                .hits()
                .iter()
                .map(|hit| hit.document_id)
                .collect::<Vec<_>>(),
            vec![10, 20]
        );
    }

    #[test]
    fn rerank_failure_does_not_publish_partial_results() {
        let initial = [ExactCandidate {
            document_id: 1,
            position: 1,
            embedding: &[1.0, 0.0],
        }];
        let mut rerank = ExactRerank::<2>::default();
        rerank
            .rerank(ExactMetric::InnerProduct, &[1.0, 0.0], &initial, 1)
            .unwrap();
        let before = rerank.clone();
        let invalid = [
            ExactCandidate {
                document_id: 2,
                position: 2,
                embedding: &[1.0, 0.0],
            },
            ExactCandidate {
                document_id: 3,
                position: 3,
                embedding: &[f32::INFINITY, 0.0],
            },
        ];
        assert!(rerank
            .rerank(ExactMetric::InnerProduct, &[1.0, 0.0], &invalid, 2)
            .is_err());
        assert_eq!(rerank, before);
    }
}
