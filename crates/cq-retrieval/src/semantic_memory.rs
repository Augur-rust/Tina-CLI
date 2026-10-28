//! Optional persistent USEARCH recall over real, identity-bound embeddings.
//!
//! The product does not construct this backend unless a concrete embedding
//! provider is supplied. Feature hashes and lexical projections are not valid
//! providers for this semantic boundary.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    ExactCandidate, ExactMetric, ExactRerank, ExternalIndexIdentity, UsearchAnnIndex,
    UsearchIndexConfig,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SEMANTIC_SCHEMA: &str = "tina.semantic-usearch-index.v1";
const MAX_SEMANTIC_DOCUMENTS: usize = 1_000_000;
const MAX_SEMANTIC_CAPACITY: usize = MAX_SEMANTIC_DOCUMENTS + 1_024;
const MAX_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;
const MAX_RERANK_CANDIDATES: usize = 256;

pub trait SemanticEmbeddingProvider {
    fn identity(&self) -> ExternalIndexIdentity;

    fn metric(&self) -> ExactMetric {
        ExactMetric::Cosine
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, SemanticMemoryError>;
}

#[derive(Debug)]
pub enum SemanticMemoryError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Backend(String),
    Provider(String),
    Invalid(String),
    Capacity(String),
}

impl fmt::Display for SemanticMemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "semantic index I/O failure: {error}"),
            Self::Json(error) => write!(formatter, "semantic index JSON failure: {error}"),
            Self::Backend(error) => write!(formatter, "semantic index backend failure: {error}"),
            Self::Provider(error) => write!(formatter, "embedding provider failure: {error}"),
            Self::Invalid(error) => write!(formatter, "invalid semantic index: {error}"),
            Self::Capacity(error) => write!(formatter, "semantic index capacity: {error}"),
        }
    }
}

impl std::error::Error for SemanticMemoryError {}

impl From<std::io::Error> for SemanticMemoryError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for SemanticMemoryError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SemanticDocument {
    document_id: u64,
    turn_id: String,
    content_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticMemoryDocument {
    pub turn_id: String,
    pub content_sha256: String,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticMemoryHit {
    pub turn_id: String,
    pub content_sha256: String,
    pub score: f32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SemanticManifestBody {
    schema: String,
    dimensions: u32,
    revision_sha256: [u8; 32],
    license_sha256: [u8; 32],
    metric: String,
    capacity: usize,
    documents: Vec<SemanticDocument>,
    index_file: String,
    index_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SemanticManifestRecord {
    body: SemanticManifestBody,
    record_sha256: String,
}

pub struct PersistentUsearchMemoryIndex<P> {
    provider: P,
    index: UsearchAnnIndex,
    directory: PathBuf,
    capacity: usize,
    documents: BTreeMap<u64, SemanticDocument>,
    manifest_path: PathBuf,
}

impl<P: SemanticEmbeddingProvider> PersistentUsearchMemoryIndex<P> {
    pub fn build(
        provider: P,
        turns: &[SemanticMemoryDocument],
        directory: &Path,
    ) -> Result<Self, SemanticMemoryError> {
        if turns.len() > MAX_SEMANTIC_DOCUMENTS {
            return Err(SemanticMemoryError::Capacity(format!(
                "{} turns exceed {MAX_SEMANTIC_DOCUMENTS}",
                turns.len()
            )));
        }
        let identity = provider.identity();
        identity
            .validate()
            .map_err(|error| SemanticMemoryError::Invalid(error.to_string()))?;
        let capacity = turns.len().saturating_add(1_024).max(1);
        let config = UsearchIndexConfig::new(identity, provider.metric(), capacity);
        let index = UsearchAnnIndex::new(config)
            .map_err(|error| SemanticMemoryError::Backend(error.to_string()))?;
        let mut documents = BTreeMap::new();
        for turn in turns {
            let document = semantic_document(turn)?;
            if documents.contains_key(&document.document_id) {
                return Err(SemanticMemoryError::Invalid(format!(
                    "semantic document key collision for turn {}",
                    turn.turn_id
                )));
            }
            let embedding = provider.embed(&turn.content)?;
            validate_provider_embedding(identity, &embedding)?;
            index
                .add(document.document_id, &embedding)
                .map_err(|error| SemanticMemoryError::Backend(error.to_string()))?;
            documents.insert(document.document_id, document);
        }
        let mut result = Self {
            provider,
            index,
            directory: directory.to_path_buf(),
            capacity,
            documents,
            manifest_path: PathBuf::new(),
        };
        result.manifest_path = result.persist_snapshot()?;
        Ok(result)
    }

    pub fn load(provider: P, manifest_path: &Path) -> Result<Self, SemanticMemoryError> {
        let manifest: SemanticManifestRecord = read_json_limited(manifest_path)?;
        validate_manifest(&manifest, provider.identity(), provider.metric())?;
        let directory = manifest_path
            .parent()
            .ok_or_else(|| SemanticMemoryError::Invalid("manifest has no parent".to_string()))?
            .to_path_buf();
        let index_path = directory.join(&manifest.body.index_file);
        if file_sha256(&index_path)? != manifest.body.index_sha256 {
            return Err(SemanticMemoryError::Invalid(
                "USEARCH index content hash drifted".to_string(),
            ));
        }
        let config = UsearchIndexConfig::new(
            provider.identity(),
            provider.metric(),
            manifest.body.capacity,
        );
        let index = UsearchAnnIndex::load(config, &index_path)
            .map_err(|error| SemanticMemoryError::Backend(error.to_string()))?;
        if index.len() != manifest.body.documents.len() {
            return Err(SemanticMemoryError::Invalid(format!(
                "USEARCH document count {} does not match manifest {}",
                index.len(),
                manifest.body.documents.len()
            )));
        }
        let documents = manifest
            .body
            .documents
            .into_iter()
            .map(|document| (document.document_id, document))
            .collect::<BTreeMap<_, _>>();
        for document_id in documents.keys() {
            index
                .get(*document_id)
                .map_err(|error| SemanticMemoryError::Invalid(error.to_string()))?;
        }
        Ok(Self {
            provider,
            index,
            directory,
            capacity: manifest.body.capacity,
            documents,
            manifest_path: manifest_path.to_path_buf(),
        })
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn add_document(
        &mut self,
        turn: &SemanticMemoryDocument,
    ) -> Result<PathBuf, SemanticMemoryError> {
        if self.documents.len() >= self.capacity {
            return Err(SemanticMemoryError::Capacity(format!(
                "semantic index reached capacity {}",
                self.capacity
            )));
        }
        let document = semantic_document(turn)?;
        if let Some(existing) = self.documents.get(&document.document_id) {
            if existing == &document {
                return Ok(self.manifest_path.clone());
            }
            return Err(SemanticMemoryError::Invalid(format!(
                "semantic document key collision for turn {}",
                turn.turn_id
            )));
        }
        let embedding = self.provider.embed(&turn.content)?;
        validate_provider_embedding(self.provider.identity(), &embedding)?;
        self.index
            .add(document.document_id, &embedding)
            .map_err(|error| SemanticMemoryError::Backend(error.to_string()))?;
        let document_id = document.document_id;
        self.documents.insert(document_id, document);
        match self.persist_snapshot() {
            Ok(manifest_path) => {
                self.manifest_path = manifest_path.clone();
                Ok(manifest_path)
            }
            Err(error) => {
                self.documents.remove(&document_id);
                match self.index.remove(document_id) {
                    Ok(true) => Err(error),
                    Ok(false) => Err(SemanticMemoryError::Backend(format!(
                        "semantic snapshot failed ({error}) and USEARCH rollback did not find document {document_id}"
                    ))),
                    Err(rollback) => Err(SemanticMemoryError::Backend(format!(
                        "semantic snapshot failed ({error}) and USEARCH rollback failed: {rollback}"
                    ))),
                }
            }
        }
    }

    fn persist_snapshot(&self) -> Result<PathBuf, SemanticMemoryError> {
        fs::create_dir_all(&self.directory)?;
        let snapshot_id = snapshot_id(self.provider.identity(), self.documents.values());
        let index_file = format!("semantic-{snapshot_id}.usearch");
        let index_path = self.directory.join(&index_file);
        if !index_path.exists() {
            let temporary = self.directory.join(format!(
                ".{index_file}.{}.{}.tmp",
                std::process::id(),
                unix_nanos()?
            ));
            let result = (|| -> Result<(), SemanticMemoryError> {
                self.index
                    .save(&temporary)
                    .map_err(|error| SemanticMemoryError::Backend(error.to_string()))?;
                fs::rename(&temporary, &index_path)?;
                Ok(())
            })();
            if result.is_err() {
                let _ = fs::remove_file(&temporary);
            }
            result?;
        }
        let index_sha256 = file_sha256(&index_path)?;
        let body = SemanticManifestBody {
            schema: SEMANTIC_SCHEMA.to_string(),
            dimensions: self.provider.identity().dimensions,
            revision_sha256: self.provider.identity().revision_sha256,
            license_sha256: self.provider.identity().license_sha256,
            metric: metric_name(self.provider.metric()).to_string(),
            capacity: self.capacity,
            documents: self.documents.values().cloned().collect(),
            index_file,
            index_sha256,
        };
        let manifest = SemanticManifestRecord {
            record_sha256: canonical_hash(&body)?,
            body,
        };
        let manifest_path = self.directory.join(format!("semantic-{snapshot_id}.json"));
        if !manifest_path.exists() {
            write_new_json(&manifest_path, &manifest)?;
        } else {
            let existing: SemanticManifestRecord = read_json_limited(&manifest_path)?;
            if existing != manifest {
                return Err(SemanticMemoryError::Invalid(format!(
                    "semantic snapshot {snapshot_id} already exists with different bytes"
                )));
            }
        }
        Ok(manifest_path)
    }

    pub fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SemanticMemoryHit>, SemanticMemoryError> {
        if limit == 0 || self.documents.is_empty() {
            return Ok(Vec::new());
        }
        let query_embedding = self.provider.embed(query)?;
        validate_provider_embedding(self.provider.identity(), &query_embedding)?;
        let ann_limit = limit
            .saturating_mul(4)
            .max(limit)
            .min(MAX_RERANK_CANDIDATES)
            .min(self.documents.len());
        let ann = self
            .index
            .search(&query_embedding, ann_limit)
            .map_err(|error| SemanticMemoryError::Backend(error.to_string()))?;
        let mut embeddings = Vec::with_capacity(ann.len());
        for candidate in &ann {
            let embedding = self
                .index
                .get(candidate.document_id)
                .map_err(|error| SemanticMemoryError::Backend(error.to_string()))?;
            embeddings.push((candidate.document_id, embedding));
        }
        let exact_candidates = embeddings
            .iter()
            .enumerate()
            .map(|(index, (document_id, embedding))| ExactCandidate {
                document_id: *document_id,
                position: u32::try_from(index + 1).expect("candidate limit fits u32"),
                embedding,
            })
            .collect::<Vec<_>>();
        let mut rerank = ExactRerank::<MAX_RERANK_CANDIDATES>::default();
        rerank
            .rerank(
                self.provider.metric(),
                &query_embedding,
                &exact_candidates,
                limit.min(MAX_RERANK_CANDIDATES),
            )
            .map_err(|error| SemanticMemoryError::Backend(error.to_string()))?;
        let mut output = Vec::with_capacity(rerank.hits().len());
        for hit in rerank.hits() {
            let document = self.documents.get(&hit.document_id).ok_or_else(|| {
                SemanticMemoryError::Invalid(format!(
                    "USEARCH returned unknown document {}",
                    hit.document_id
                ))
            })?;
            output.push(SemanticMemoryHit {
                turn_id: document.turn_id.clone(),
                content_sha256: document.content_sha256.clone(),
                score: hit.score,
            });
        }
        Ok(output)
    }
}

impl<P: SemanticEmbeddingProvider> PersistentUsearchMemoryIndex<P> {
    pub fn backend_name(&self) -> &'static str {
        "usearch-real-embedding"
    }
}

fn semantic_document(
    turn: &SemanticMemoryDocument,
) -> Result<SemanticDocument, SemanticMemoryError> {
    if !is_lower_sha256(&turn.turn_id) {
        return Err(SemanticMemoryError::Invalid(
            "turn id is not a lower-case SHA-256 hex string".to_string(),
        ));
    }
    if turn.content_sha256 != format!("{:x}", Sha256::digest(turn.content.as_bytes())) {
        return Err(SemanticMemoryError::Invalid(format!(
            "turn {} content hash drifted",
            turn.turn_id
        )));
    }
    let prefix = turn
        .turn_id
        .get(..16)
        .ok_or_else(|| SemanticMemoryError::Invalid("turn id is too short".to_string()))?;
    let document_id = u64::from_str_radix(prefix, 16)
        .map_err(|_| SemanticMemoryError::Invalid("turn id is not hexadecimal".to_string()))?;
    Ok(SemanticDocument {
        document_id,
        turn_id: turn.turn_id.clone(),
        content_sha256: turn.content_sha256.clone(),
    })
}

fn validate_provider_embedding(
    identity: ExternalIndexIdentity,
    embedding: &[f32],
) -> Result<(), SemanticMemoryError> {
    if embedding.len() != identity.dimensions as usize {
        return Err(SemanticMemoryError::Provider(format!(
            "provider returned {} dimensions, expected {}",
            embedding.len(),
            identity.dimensions
        )));
    }
    if let Some(index) = embedding.iter().position(|value| !value.is_finite()) {
        return Err(SemanticMemoryError::Provider(format!(
            "provider returned non-finite dimension {index}"
        )));
    }
    Ok(())
}

fn validate_manifest(
    manifest: &SemanticManifestRecord,
    identity: ExternalIndexIdentity,
    metric: ExactMetric,
) -> Result<(), SemanticMemoryError> {
    let body = &manifest.body;
    if body.schema != SEMANTIC_SCHEMA
        || body.dimensions != identity.dimensions
        || body.revision_sha256 != identity.revision_sha256
        || body.license_sha256 != identity.license_sha256
        || body.metric != metric_name(metric)
        || body.capacity == 0
        || body.capacity > MAX_SEMANTIC_CAPACITY
        || body.documents.len() > body.capacity
        || !is_lower_sha256(&body.index_sha256)
        || !valid_index_file_name(&body.index_file)
    {
        return Err(SemanticMemoryError::Invalid(
            "semantic manifest identity or capacity drifted".to_string(),
        ));
    }
    if manifest.record_sha256 != canonical_hash(body)? {
        return Err(SemanticMemoryError::Invalid(
            "semantic manifest record hash drifted".to_string(),
        ));
    }
    let mut keys = BTreeSet::new();
    let mut turns = BTreeSet::new();
    for document in &body.documents {
        let expected_key = document
            .turn_id
            .get(..16)
            .filter(|_| {
                is_lower_sha256(&document.turn_id) && is_lower_sha256(&document.content_sha256)
            })
            .and_then(|prefix| u64::from_str_radix(prefix, 16).ok());
        if expected_key != Some(document.document_id)
            || !keys.insert(document.document_id)
            || !turns.insert(&document.turn_id)
        {
            return Err(SemanticMemoryError::Invalid(
                "semantic manifest contains invalid or duplicate documents".to_string(),
            ));
        }
    }
    Ok(())
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn valid_index_file_name(value: &str) -> bool {
    let path = Path::new(value);
    let Some(snapshot) = value
        .strip_prefix("semantic-")
        .and_then(|value| value.strip_suffix(".usearch"))
    else {
        return false;
    };
    is_lower_sha256(snapshot)
        && !path.is_absolute()
        && path
            .parent()
            .is_some_and(|parent| parent.as_os_str().is_empty())
        && path.file_name().and_then(|name| name.to_str()) == Some(value)
}

fn snapshot_id<'a>(
    identity: ExternalIndexIdentity,
    documents: impl Iterator<Item = &'a SemanticDocument>,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"tina-semantic-snapshot-v1\0");
    digest.update(identity.dimensions.to_le_bytes());
    digest.update(identity.revision_sha256);
    digest.update(identity.license_sha256);
    for document in documents {
        digest.update(document.document_id.to_le_bytes());
        digest.update(document.turn_id.as_bytes());
        digest.update(document.content_sha256.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn metric_name(metric: ExactMetric) -> &'static str {
    match metric {
        ExactMetric::Cosine => "cosine",
        ExactMetric::InnerProduct => "inner_product",
        ExactMetric::SquaredL2 => "squared_l2",
    }
}

fn canonical_hash(value: &impl Serialize) -> Result<String, SemanticMemoryError> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

fn file_sha256(path: &Path) -> Result<String, SemanticMemoryError> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_new_json(path: &Path, value: &impl Serialize) -> Result<(), SemanticMemoryError> {
    let temporary =
        path.with_extension(format!("json.{}.{}.tmp", std::process::id(), unix_nanos()?));
    let result = (|| -> Result<(), SemanticMemoryError> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        serde_json::to_writer(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn read_json_limited<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, SemanticMemoryError> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() > MAX_MANIFEST_BYTES {
        return Err(SemanticMemoryError::Capacity(format!(
            "semantic manifest {} exceeds {MAX_MANIFEST_BYTES} bytes",
            path.display()
        )));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_MANIFEST_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(SemanticMemoryError::Capacity(format!(
            "semantic manifest {} exceeds {MAX_MANIFEST_BYTES} bytes",
            path.display()
        )));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn unix_nanos() -> Result<u128, SemanticMemoryError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| SemanticMemoryError::Invalid(format!("clock before epoch: {error}")))?
        .as_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct FixtureProvider;

    impl SemanticEmbeddingProvider for FixtureProvider {
        fn identity(&self) -> ExternalIndexIdentity {
            ExternalIndexIdentity {
                dimensions: 3,
                revision_sha256: [7; 32],
                license_sha256: [9; 32],
            }
        }

        fn embed(&self, text: &str) -> Result<Vec<f32>, SemanticMemoryError> {
            let lower = text.to_lowercase();
            Ok(vec![
                lower.matches("rust").count() as f32,
                lower.matches("cuda").count() as f32,
                lower.matches("python").count() as f32,
            ])
        }
    }

    fn document(seed: &str, content: &str) -> SemanticMemoryDocument {
        SemanticMemoryDocument {
            turn_id: format!("{:x}", Sha256::digest(seed.as_bytes())),
            content_sha256: format!("{:x}", Sha256::digest(content.as_bytes())),
            content: content.to_string(),
        }
    }

    #[test]
    fn real_provider_boundary_persists_and_exactly_reranks_usearch_candidates() {
        let root = std::env::temp_dir().join(format!(
            "tina-semantic-test-{}-{}",
            std::process::id(),
            unix_nanos().unwrap()
        ));
        fs::create_dir(&root).unwrap();
        let rust = document("rust-turn", "Rust ownership and lifetimes");
        let cuda = document("cuda-turn", "CUDA kernel launch details");
        let index = PersistentUsearchMemoryIndex::build(
            FixtureProvider,
            &[rust.clone(), cuda],
            &root.join("index"),
        )
        .unwrap();
        let hits = index.search("Rust borrow checking", 1).unwrap();
        assert_eq!(hits[0].turn_id, rust.turn_id);
        assert_eq!(hits[0].content_sha256, rust.content_sha256);

        let mut loaded =
            PersistentUsearchMemoryIndex::load(FixtureProvider, index.manifest_path()).unwrap();
        assert_eq!(loaded.search("Rust", 1).unwrap()[0].turn_id, rust.turn_id);
        let python = document("python-turn", "Python packaging details");
        let manifest = loaded.add_document(&python).unwrap();
        let reloaded = PersistentUsearchMemoryIndex::load(FixtureProvider, &manifest).unwrap();
        assert_eq!(
            reloaded.search("Python", 1).unwrap()[0].turn_id,
            python.turn_id
        );
        assert_eq!(loaded.backend_name(), "usearch-real-embedding");
        assert!(root.starts_with(std::env::temp_dir()));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn document_hashes_and_snapshot_paths_fail_closed() {
        let snapshot = "a".repeat(64);
        assert!(valid_index_file_name(&format!(
            "semantic-{snapshot}.usearch"
        )));
        assert!(!valid_index_file_name(&format!(
            "../semantic-{snapshot}.usearch"
        )));
        assert!(!valid_index_file_name("semantic-short.usearch"));

        let mut invalid = document("invalid-turn", "stable content");
        invalid.content.push('!');
        let error = semantic_document(&invalid).unwrap_err();
        assert!(error.to_string().contains("content hash drifted"));
    }
}
