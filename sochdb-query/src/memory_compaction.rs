// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Hierarchical Memory Compaction (Task 5)
//!
//! This module implements semantic memory compaction inspired by LSM-trees.
//! It manages tiered storage where older memories are summarized to maintain
//! bounded context while preserving semantic continuity.
//!
//! ## Architecture
//!
//! ```text
//! L0: Raw Episodes (recent, full detail)
//!     │
//!     ▼ Summarization
//! L1: Summaries (older, compressed)
//!     │
//!     ▼ Abstraction
//! L2: Abstractions (oldest, highly compressed)
//! ```
//!
//! ## Compaction Strategy
//!
//! - Episodes older than tier threshold are grouped by semantic similarity
//! - Each group is summarized (via LLM or extractive methods)
//! - Summaries are re-embedded for retrieval
//! - Growth is O(log_c T) where c = compaction ratio, T = total events

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for memory compaction
#[derive(Debug, Clone)]
pub struct MemoryCompactionConfig {
    /// Maximum episodes in L0 before compaction
    pub l0_max_episodes: usize,
    
    /// Maximum summaries in L1 before compaction
    pub l1_max_summaries: usize,
    
    /// Age threshold for L0 → L1 compaction (seconds)
    pub l0_age_threshold_secs: u64,
    
    /// Age threshold for L1 → L2 compaction (seconds)
    pub l1_age_threshold_secs: u64,
    
    /// Number of episodes to group for summarization
    pub group_size: usize,
    
    /// Similarity threshold for grouping (0.0 to 1.0)
    pub similarity_threshold: f32,
    
    /// Maximum tokens per summary
    pub max_summary_tokens: usize,
    
    /// Whether to re-embed summaries for retrieval
    pub reembed_summaries: bool,
    
    /// Compaction check interval (seconds)
    pub check_interval_secs: u64,
}

impl Default for MemoryCompactionConfig {
    fn default() -> Self {
        Self {
            l0_max_episodes: 1000,
            l1_max_summaries: 100,
            l0_age_threshold_secs: 3600,      // 1 hour
            l1_age_threshold_secs: 86400 * 7, // 1 week
            group_size: 10,
            similarity_threshold: 0.7,
            max_summary_tokens: 200,
            reembed_summaries: true,
            check_interval_secs: 300, // 5 minutes
        }
    }
}

impl MemoryCompactionConfig {
    /// Create config for aggressive compaction (testing/demos)
    pub fn aggressive() -> Self {
        Self {
            l0_max_episodes: 100,
            l1_max_summaries: 20,
            l0_age_threshold_secs: 60,
            l1_age_threshold_secs: 3600,
            group_size: 5,
            ..Default::default()
        }
    }
    
    /// Create config for long-running agents
    pub fn long_running() -> Self {
        Self {
            l0_max_episodes: 5000,
            l1_max_summaries: 500,
            l0_age_threshold_secs: 3600 * 6,      // 6 hours
            l1_age_threshold_secs: 86400 * 30,    // 30 days
            group_size: 20,
            ..Default::default()
        }
    }
}

// ============================================================================
// Memory Types
// ============================================================================

/// A raw episode (L0)
#[derive(Debug, Clone)]
pub struct Episode {
    /// Unique identifier
    pub id: String,
    
    /// Timestamp (seconds since epoch)
    pub timestamp: f64,
    
    /// Episode content (e.g., user message, tool call)
    pub content: String,
    
    /// Episode type
    pub episode_type: EpisodeType,
    
    /// Associated metadata
    pub metadata: HashMap<String, String>,
    
    /// Embedding vector (for similarity grouping)
    pub embedding: Option<Vec<f32>>,
    
    /// Token count (estimated or exact)
    pub token_count: usize,
}

/// Episode types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpisodeType {
    /// User message
    UserMessage,
    /// Assistant response
    AssistantResponse,
    /// Tool call
    ToolCall,
    /// Tool result
    ToolResult,
    /// System event
    SystemEvent,
    /// Observation
    Observation,
}

/// A summary (L1)
#[derive(Debug, Clone)]
pub struct Summary {
    /// Unique identifier
    pub id: String,
    
    /// Summarized content
    pub content: String,
    
    /// IDs of episodes that were summarized
    pub source_episode_ids: Vec<String>,
    
    /// Time range covered
    pub time_range: (f64, f64),
    
    /// Summary embedding
    pub embedding: Option<Vec<f32>>,
    
    /// Token count
    pub token_count: usize,
    
    /// When this summary was created
    pub created_at: f64,
    
    /// Topics/themes extracted
    pub topics: Vec<String>,
}

/// An abstraction (L2)
#[derive(Debug, Clone)]
pub struct Abstraction {
    /// Unique identifier
    pub id: String,
    
    /// High-level abstraction content
    pub content: String,
    
    /// IDs of summaries that were abstracted
    pub source_summary_ids: Vec<String>,
    
    /// Time range covered
    pub time_range: (f64, f64),
    
    /// Abstraction embedding
    pub embedding: Option<Vec<f32>>,
    
    /// Token count
    pub token_count: usize,
    
    /// When this abstraction was created
    pub created_at: f64,
    
    /// Key insights
    pub insights: Vec<String>,
}

// ============================================================================
// Summarizer Trait
// ============================================================================

/// Trait for summarization backends
pub trait Summarizer: Send + Sync {
    /// Summarize a group of episodes into a single summary
    fn summarize_episodes(&self, episodes: &[Episode]) -> Result<String, CompactionError>;
    
    /// Summarize a group of summaries into an abstraction
    fn abstract_summaries(&self, summaries: &[Summary]) -> Result<String, CompactionError>;
    
    /// Extract topics/themes from content
    fn extract_topics(&self, content: &str) -> Vec<String>;
}

/// Extractive summarizer (no LLM required)
pub struct ExtractiveSummarizer {
    /// Maximum sentences to include
    pub max_sentences: usize,
    
    /// Whether to include timestamps
    pub include_timestamps: bool,
}

impl Default for ExtractiveSummarizer {
    fn default() -> Self {
        Self {
            max_sentences: 5,
            include_timestamps: true,
        }
    }
}

impl Summarizer for ExtractiveSummarizer {
    fn summarize_episodes(&self, episodes: &[Episode]) -> Result<String, CompactionError> {
        if episodes.is_empty() {
            return Ok(String::new());
        }
        
        let mut summary_parts = Vec::new();
        
        // Time range
        let first_ts = episodes.first().map(|e| e.timestamp).unwrap_or(0.0);
        let last_ts = episodes.last().map(|e| e.timestamp).unwrap_or(0.0);
        
        if self.include_timestamps {
            summary_parts.push(format!(
                "[{} episodes over {:.0} seconds]",
                episodes.len(),
                last_ts - first_ts
            ));
        }
        
        // Group by type and summarize
        let mut by_type: HashMap<EpisodeType, Vec<&Episode>> = HashMap::new();
        for episode in episodes {
            by_type.entry(episode.episode_type).or_default().push(episode);
        }
        
        // Extract key content from each type
        for (ep_type, eps) in by_type {
            let type_name = match ep_type {
                EpisodeType::UserMessage => "User messages",
                EpisodeType::AssistantResponse => "Responses",
                EpisodeType::ToolCall => "Tool calls",
                EpisodeType::ToolResult => "Tool results",
                EpisodeType::SystemEvent => "Events",
                EpisodeType::Observation => "Observations",
            };
            
            // Take first sentence from each, up to max_sentences
            let sentences: Vec<String> = eps
                .iter()
                .take(self.max_sentences)
                .filter_map(|e| e.content.split('.').next().map(|s| s.trim().to_string()))
                .filter(|s| !s.is_empty())
                .collect();
            
            if !sentences.is_empty() {
                summary_parts.push(format!("{}: {}", type_name, sentences.join("; ")));
            }
        }
        
        Ok(summary_parts.join("\n"))
    }
    
    fn abstract_summaries(&self, summaries: &[Summary]) -> Result<String, CompactionError> {
        if summaries.is_empty() {
            return Ok(String::new());
        }
        
        let mut abstraction_parts = Vec::new();
        
        // Time range
        let first_ts = summaries.iter().map(|s| s.time_range.0).fold(f64::MAX, f64::min);
        let last_ts = summaries.iter().map(|s| s.time_range.1).fold(f64::MIN, f64::max);
        
        abstraction_parts.push(format!(
            "[{} summaries, {:.1} hours span]",
            summaries.len(),
            (last_ts - first_ts) / 3600.0
        ));
        
        // Collect all topics
        let all_topics: Vec<&str> = summaries
            .iter()
            .flat_map(|s| s.topics.iter().map(|t| t.as_str()))
            .collect();
        
        if !all_topics.is_empty() {
            let unique_topics: Vec<_> = all_topics.iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .take(10)
                .collect();
            abstraction_parts.push(format!("Topics: {}", unique_topics.join(", ")));
        }
        
        // Take first line of each summary
        let key_points: Vec<String> = summaries
            .iter()
            .take(5)
            .filter_map(|s| s.content.lines().next().map(|l| l.to_string()))
            .collect();
        
        if !key_points.is_empty() {
            abstraction_parts.push(format!("Key points:\n- {}", key_points.join("\n- ")));
        }
        
        Ok(abstraction_parts.join("\n"))
    }
    
    fn extract_topics(&self, content: &str) -> Vec<String> {
        // Simple keyword extraction (would use NLP in production)
        let stopwords = ["the", "a", "an", "is", "are", "was", "were", "to", "from", "in", "on", "at", "for", "and", "or"];
        
        let words: Vec<&str> = content
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .filter(|w| !stopwords.contains(&w.to_lowercase().as_str()))
            .collect();
        
        // Count word frequencies
        let mut freq: HashMap<String, usize> = HashMap::new();
        for word in words {
            let normalized = word.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if normalized.len() > 3 {
                *freq.entry(normalized).or_insert(0) += 1;
            }
        }
        
        // Return top 5 by frequency
        let mut sorted: Vec<_> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        
        sorted.into_iter().take(5).map(|(w, _)| w).collect()
    }
}

// ============================================================================
// Memory Store
// ============================================================================

/// Hierarchical memory store with compaction
pub struct HierarchicalMemory<S: Summarizer> {
    /// Configuration
    config: MemoryCompactionConfig,
    
    /// L0: Raw episodes
    l0_episodes: RwLock<VecDeque<Episode>>,
    
    /// L1: Summaries
    l1_summaries: RwLock<VecDeque<Summary>>,
    
    /// L2: Abstractions
    l2_abstractions: RwLock<VecDeque<Abstraction>>,
    
    /// Summarizer backend
    summarizer: Arc<S>,
    
    /// Compaction statistics
    stats: RwLock<CompactionStats>,
    
    /// ID counter
    next_id: std::sync::atomic::AtomicU64,
}

/// Compaction statistics
#[derive(Debug, Clone, Default)]
pub struct CompactionStats {
    /// Total episodes added
    pub total_episodes: usize,
    
    /// Total summaries created
    pub total_summaries: usize,
    
    /// Total abstractions created
    pub total_abstractions: usize,
    
    /// Episodes compacted (removed from L0)
    pub episodes_compacted: usize,
    
    /// Summaries compacted (removed from L1)
    pub summaries_compacted: usize,
    
    /// Last compaction time
    pub last_compaction: Option<f64>,
    
    /// Total token savings (estimated)
    pub token_savings: usize,
}

impl<S: Summarizer> HierarchicalMemory<S> {
    /// Create a new hierarchical memory store
    pub fn new(config: MemoryCompactionConfig, summarizer: Arc<S>) -> Self {
        Self {
            config,
            l0_episodes: RwLock::new(VecDeque::new()),
            l1_summaries: RwLock::new(VecDeque::new()),
            l2_abstractions: RwLock::new(VecDeque::new()),
            summarizer,
            stats: RwLock::new(CompactionStats::default()),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }
    
    /// Generate next ID
    fn next_id(&self) -> String {
        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("mem_{}", id)
    }
    
    /// Add an episode to L0
    pub fn add_episode(&self, content: String, episode_type: EpisodeType) -> String {
        let id = self.next_id();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        let token_count = content.len() / 4; // Rough estimate
        
        let episode = Episode {
            id: id.clone(),
            timestamp,
            content,
            episode_type,
            metadata: HashMap::new(),
            embedding: None,
            token_count,
        };
        
        {
            let mut l0 = self.l0_episodes.write().unwrap();
            l0.push_back(episode);
        }
        
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_episodes += 1;
        }
        
        id
    }
    
    /// Add episode with embedding
    pub fn add_episode_with_embedding(
        &self,
        content: String,
        episode_type: EpisodeType,
        embedding: Vec<f32>,
    ) -> String {
        let id = self.next_id();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        let token_count = content.len() / 4;
        
        let episode = Episode {
            id: id.clone(),
            timestamp,
            content,
            episode_type,
            metadata: HashMap::new(),
            embedding: Some(embedding),
            token_count,
        };
        
        {
            let mut l0 = self.l0_episodes.write().unwrap();
            l0.push_back(episode);
        }
        
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_episodes += 1;
        }
        
        id
    }
    
    /// Check if compaction is needed and run if so
    pub fn maybe_compact(&self) -> Result<bool, CompactionError> {
        let needs_l0 = {
            let l0 = self.l0_episodes.read().unwrap();
            l0.len() >= self.config.l0_max_episodes
        };
        
        let needs_l1 = {
            let l1 = self.l1_summaries.read().unwrap();
            l1.len() >= self.config.l1_max_summaries
        };
        
        if needs_l0 || needs_l1 {
            self.run_compaction()?;
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Run compaction cycle
    pub fn run_compaction(&self) -> Result<(), CompactionError> {
        // L0 → L1 compaction
        self.compact_l0_to_l1()?;
        
        // L1 → L2 compaction
        self.compact_l1_to_l2()?;
        
        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.last_compaction = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64()
            );
        }
        
        Ok(())
    }
    
    /// Compact L0 episodes to L1 summaries
    fn compact_l0_to_l1(&self) -> Result<(), CompactionError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        let age_threshold = now - self.config.l0_age_threshold_secs as f64;
        
        // Collect episodes to compact
        let to_compact: Vec<Episode> = {
            let l0 = self.l0_episodes.read().unwrap();
            l0.iter()
                .filter(|e| e.timestamp < age_threshold)
                .cloned()
                .collect()
        };
        
        if to_compact.is_empty() {
            return Ok(());
        }
        
        // Group episodes by similarity or time
        let groups = self.group_episodes(&to_compact);
        
        // Summarize each group
        for group in groups {
            if group.is_empty() {
                continue;
            }
            
            let content = self.summarizer.summarize_episodes(&group)?;
            let topics = self.summarizer.extract_topics(&content);
            
            let first_ts = group.iter().map(|e| e.timestamp).fold(f64::MAX, f64::min);
            let last_ts = group.iter().map(|e| e.timestamp).fold(f64::MIN, f64::max);
            
            let episode_ids: Vec<String> = group.iter().map(|e| e.id.clone()).collect();
            let original_tokens: usize = group.iter().map(|e| e.token_count).sum();
            let summary_tokens = content.len() / 4;
            
            let summary = Summary {
                id: self.next_id(),
                content,
                source_episode_ids: episode_ids,
                time_range: (first_ts, last_ts),
                embedding: None, // Would be generated if reembed_summaries is true
                token_count: summary_tokens,
                created_at: now,
                topics,
            };
            
            // Add summary to L1
            {
                let mut l1 = self.l1_summaries.write().unwrap();
                l1.push_back(summary);
            }
            
            // Update stats
            {
                let mut stats = self.stats.write().unwrap();
                stats.total_summaries += 1;
                stats.episodes_compacted += group.len();
                stats.token_savings += original_tokens.saturating_sub(summary_tokens);
            }
        }
        
        // Remove compacted episodes from L0
        {
            let mut l0 = self.l0_episodes.write().unwrap();
            l0.retain(|e| e.timestamp >= age_threshold);
        }
        
        Ok(())
    }
    
    /// Compact L1 summaries to L2 abstractions
    fn compact_l1_to_l2(&self) -> Result<(), CompactionError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        let age_threshold = now - self.config.l1_age_threshold_secs as f64;
        
        // Collect summaries to compact
        let to_compact: Vec<Summary> = {
            let l1 = self.l1_summaries.read().unwrap();
            l1.iter()
                .filter(|s| s.created_at < age_threshold)
                .cloned()
                .collect()
        };
        
        if to_compact.len() < self.config.group_size {
            return Ok(());
        }
        
        // Group summaries
        let groups = self.group_summaries(&to_compact);
        
        for group in groups {
            if group.is_empty() {
                continue;
            }
            
            let content = self.summarizer.abstract_summaries(&group)?;
            
            let first_ts = group.iter().map(|s| s.time_range.0).fold(f64::MAX, f64::min);
            let last_ts = group.iter().map(|s| s.time_range.1).fold(f64::MIN, f64::max);
            
            let summary_ids: Vec<String> = group.iter().map(|s| s.id.clone()).collect();
            let original_tokens: usize = group.iter().map(|s| s.token_count).sum();
            let abstraction_tokens = content.len() / 4;
            
            // Extract insights from topics
            let insights: Vec<String> = group
                .iter()
                .flat_map(|s| s.topics.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .take(5)
                .collect();
            
            let abstraction = Abstraction {
                id: self.next_id(),
                content,
                source_summary_ids: summary_ids,
                time_range: (first_ts, last_ts),
                embedding: None,
                token_count: abstraction_tokens,
                created_at: now,
                insights,
            };
            
            // Add abstraction to L2
            {
                let mut l2 = self.l2_abstractions.write().unwrap();
                l2.push_back(abstraction);
            }
            
            // Update stats
            {
                let mut stats = self.stats.write().unwrap();
                stats.total_abstractions += 1;
                stats.summaries_compacted += group.len();
                stats.token_savings += original_tokens.saturating_sub(abstraction_tokens);
            }
        }
        
        // Remove compacted summaries from L1
        {
            let mut l1 = self.l1_summaries.write().unwrap();
            l1.retain(|s| s.created_at >= age_threshold);
        }
        
        Ok(())
    }
    
    /// Group episodes by time windows (simplified)
    fn group_episodes(&self, episodes: &[Episode]) -> Vec<Vec<Episode>> {
        // Simple grouping by fixed size
        episodes
            .chunks(self.config.group_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
    
    /// Group summaries by time windows
    fn group_summaries(&self, summaries: &[Summary]) -> Vec<Vec<Summary>> {
        summaries
            .chunks(self.config.group_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
    
    /// Get total token count across all tiers
    pub fn total_tokens(&self) -> usize {
        let l0: usize = self.l0_episodes.read().unwrap().iter().map(|e| e.token_count).sum();
        let l1: usize = self.l1_summaries.read().unwrap().iter().map(|s| s.token_count).sum();
        let l2: usize = self.l2_abstractions.read().unwrap().iter().map(|a| a.token_count).sum();
        
        l0 + l1 + l2
    }
    
    /// Get memory for context assembly (most recent first)
    pub fn get_context(&self, max_tokens: usize) -> Vec<MemoryEntry> {
        let mut entries = Vec::new();
        let mut tokens_used = 0;
        
        // Start with L0 (most recent)
        let l0 = self.l0_episodes.read().unwrap();
        for episode in l0.iter().rev() {
            if tokens_used + episode.token_count > max_tokens {
                break;
            }
            entries.push(MemoryEntry::Episode(episode.clone()));
            tokens_used += episode.token_count;
        }
        
        // Add L1 summaries if space
        let l1 = self.l1_summaries.read().unwrap();
        for summary in l1.iter().rev() {
            if tokens_used + summary.token_count > max_tokens {
                break;
            }
            entries.push(MemoryEntry::Summary(summary.clone()));
            tokens_used += summary.token_count;
        }
        
        // Add L2 abstractions if space
        let l2 = self.l2_abstractions.read().unwrap();
        for abstraction in l2.iter().rev() {
            if tokens_used + abstraction.token_count > max_tokens {
                break;
            }
            entries.push(MemoryEntry::Abstraction(abstraction.clone()));
            tokens_used += abstraction.token_count;
        }
        
        entries
    }
    
    /// Get statistics
    pub fn stats(&self) -> CompactionStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Get tier counts
    pub fn tier_counts(&self) -> (usize, usize, usize) {
        let l0 = self.l0_episodes.read().unwrap().len();
        let l1 = self.l1_summaries.read().unwrap().len();
        let l2 = self.l2_abstractions.read().unwrap().len();
        (l0, l1, l2)
    }
}

/// Entry from hierarchical memory
#[derive(Debug, Clone)]
pub enum MemoryEntry {
    Episode(Episode),
    Summary(Summary),
    Abstraction(Abstraction),
}

impl MemoryEntry {
    /// Get content
    pub fn content(&self) -> &str {
        match self {
            Self::Episode(e) => &e.content,
            Self::Summary(s) => &s.content,
            Self::Abstraction(a) => &a.content,
        }
    }
    
    /// Get token count
    pub fn token_count(&self) -> usize {
        match self {
            Self::Episode(e) => e.token_count,
            Self::Summary(s) => s.token_count,
            Self::Abstraction(a) => a.token_count,
        }
    }
    
    /// Get tier level
    pub fn tier(&self) -> usize {
        match self {
            Self::Episode(_) => 0,
            Self::Summary(_) => 1,
            Self::Abstraction(_) => 2,
        }
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Compaction error
#[derive(Debug, Clone)]
pub enum CompactionError {
    /// Summarization failed
    SummarizationFailed(String),
    /// Embedding failed
    EmbeddingFailed(String),
    /// Storage error
    StorageError(String),
}

impl std::fmt::Display for CompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SummarizationFailed(msg) => write!(f, "Summarization failed: {}", msg),
            Self::EmbeddingFailed(msg) => write!(f, "Embedding failed: {}", msg),
            Self::StorageError(msg) => write!(f, "Storage error: {}", msg),
        }
    }
}

impl std::error::Error for CompactionError {}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a hierarchical memory with extractive summarizer
pub fn create_hierarchical_memory() -> HierarchicalMemory<ExtractiveSummarizer> {
    HierarchicalMemory::new(
        MemoryCompactionConfig::default(),
        Arc::new(ExtractiveSummarizer::default()),
    )
}

/// Create with aggressive compaction for testing
pub fn create_test_memory() -> HierarchicalMemory<ExtractiveSummarizer> {
    HierarchicalMemory::new(
        MemoryCompactionConfig::aggressive(),
        Arc::new(ExtractiveSummarizer::default()),
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_add_episode() {
        let memory = create_test_memory();
        
        let id = memory.add_episode(
            "User asked about weather".to_string(),
            EpisodeType::UserMessage,
        );
        
        assert!(id.starts_with("mem_"));
        
        let (l0, l1, l2) = memory.tier_counts();
        assert_eq!(l0, 1);
        assert_eq!(l1, 0);
        assert_eq!(l2, 0);
    }
    
    #[test]
    fn test_extractive_summarizer() {
        let summarizer = ExtractiveSummarizer::default();
        
        let episodes = vec![
            Episode {
                id: "1".to_string(),
                timestamp: 0.0,
                content: "User asked about the weather forecast.".to_string(),
                episode_type: EpisodeType::UserMessage,
                metadata: HashMap::new(),
                embedding: None,
                token_count: 10,
            },
            Episode {
                id: "2".to_string(),
                timestamp: 1.0,
                content: "Assistant provided weather information for NYC.".to_string(),
                episode_type: EpisodeType::AssistantResponse,
                metadata: HashMap::new(),
                embedding: None,
                token_count: 12,
            },
        ];
        
        let summary = summarizer.summarize_episodes(&episodes).unwrap();
        
        assert!(!summary.is_empty());
        assert!(summary.contains("episodes") || summary.contains("User") || summary.contains("Responses"));
    }
    
    #[test]
    fn test_topic_extraction() {
        let summarizer = ExtractiveSummarizer::default();
        
        let content = "The weather forecast shows sunny conditions with temperatures around 75 degrees. Tomorrow expects rain and thunderstorms across the region.";
        
        let topics = summarizer.extract_topics(content);
        
        assert!(!topics.is_empty());
        // Should extract meaningful words like "weather", "forecast", "temperatures", etc.
    }
    
    #[test]
    fn test_memory_context_retrieval() {
        let memory = create_test_memory();
        
        // Add some episodes
        for i in 0..5 {
            memory.add_episode(
                format!("Episode {} content here with some text.", i),
                EpisodeType::UserMessage,
            );
        }
        
        let context = memory.get_context(1000);
        
        assert!(!context.is_empty());
        
        // All should be L0 episodes
        for entry in &context {
            assert_eq!(entry.tier(), 0);
        }
    }
    
    #[test]
    fn test_token_tracking() {
        let memory = create_test_memory();
        
        memory.add_episode(
            "Short message".to_string(),
            EpisodeType::UserMessage,
        );
        
        memory.add_episode(
            "A much longer message with more content that should have more tokens estimated".to_string(),
            EpisodeType::AssistantResponse,
        );
        
        let total = memory.total_tokens();
        assert!(total > 0);
    }
    
    #[test]
    fn test_stats_tracking() {
        let memory = create_test_memory();
        
        for _ in 0..10 {
            memory.add_episode("Test episode".to_string(), EpisodeType::UserMessage);
        }
        
        let stats = memory.stats();
        assert_eq!(stats.total_episodes, 10);
    }
}
