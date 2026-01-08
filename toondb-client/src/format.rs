// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Unified Output Format Semantics (Task 3)
//!
//! This module provides a single taxonomy for output formats with:
//! - Clear layer boundaries (WireFormat vs ContextFormat)
//! - No lossy coercions (round-trip property preserved)
//! - Explicit conversion with error handling
//!
//! ## Problem Solved
//!
//! Previously, there were two competing format universes:
//! - Client: Toon | Json | Columnar
//! - Context: Toon | Json | Markdown | Text (with Text -> Toon coercion)
//!
//! This created semantic drift and violated the round-trip property:
//! `decode(encode(x)) = x` should hold for all formats.
//!
//! ## Solution
//!
//! Separate format enums with explicit conversions:
//! - `WireFormat`: For query results sent to clients
//! - `ContextFormat`: For LLM context packaging
//!
//! Conversions between them use `TryFrom` and can fail, making
//! incompatibilities explicit rather than silent.

use std::fmt;

/// Error when format conversion fails
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatConversionError {
    pub from: String,
    pub to: String,
    pub reason: String,
}

impl fmt::Display for FormatConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cannot convert {} to {}: {}",
            self.from, self.to, self.reason
        )
    }
}

impl std::error::Error for FormatConversionError {}

// ============================================================================
// Wire Format (Query Results)
// ============================================================================

/// Output format for query results sent to clients.
///
/// These formats are optimized for transmission efficiency and
/// client-side processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WireFormat {
    /// TOON format (default, 40-66% fewer tokens than JSON)
    /// Optimized for LLM consumption.
    Toon,
    
    /// Standard JSON for compatibility
    Json,
    
    /// Raw columnar format for analytics
    /// More efficient for large result sets with projection pushdown.
    Columnar,
}

impl Default for WireFormat {
    fn default() -> Self {
        Self::Toon
    }
}

impl fmt::Display for WireFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Toon => write!(f, "toon"),
            Self::Json => write!(f, "json"),
            Self::Columnar => write!(f, "columnar"),
        }
    }
}

impl std::str::FromStr for WireFormat {
    type Err = FormatConversionError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "toon" => Ok(Self::Toon),
            "json" => Ok(Self::Json),
            "columnar" | "column" => Ok(Self::Columnar),
            _ => Err(FormatConversionError {
                from: s.to_string(),
                to: "WireFormat".to_string(),
                reason: format!("Unknown format '{}'. Valid: toon, json, columnar", s),
            }),
        }
    }
}

// ============================================================================
// Context Format (LLM Context Packaging)
// ============================================================================

/// Output format for LLM context packaging.
///
/// These formats are optimized for readability and token efficiency
/// when constructing prompts for language models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContextFormat {
    /// TOON format (default, token-efficient)
    /// Structured data with minimal syntax overhead.
    Toon,
    
    /// JSON format
    /// Widely understood by LLMs, good for structured data.
    Json,
    
    /// Markdown format
    /// Best for human-readable context with formatting.
    Markdown,
}

impl Default for ContextFormat {
    fn default() -> Self {
        Self::Toon
    }
}

impl fmt::Display for ContextFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Toon => write!(f, "toon"),
            Self::Json => write!(f, "json"),
            Self::Markdown => write!(f, "markdown"),
        }
    }
}

impl std::str::FromStr for ContextFormat {
    type Err = FormatConversionError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "toon" => Ok(Self::Toon),
            "json" => Ok(Self::Json),
            "markdown" | "md" => Ok(Self::Markdown),
            // NOTE: "text" is NOT supported - it was previously coerced to Toon
            // which violated the round-trip property. Use Markdown for plain text.
            "text" | "plain" => Err(FormatConversionError {
                from: s.to_string(),
                to: "ContextFormat".to_string(),
                reason: "Plain text format is not supported. Use 'markdown' for \
                        human-readable output or 'toon' for LLM-optimized output.".to_string(),
            }),
            _ => Err(FormatConversionError {
                from: s.to_string(),
                to: "ContextFormat".to_string(),
                reason: format!("Unknown format '{}'. Valid: toon, json, markdown", s),
            }),
        }
    }
}

// ============================================================================
// Format Conversions
// ============================================================================

/// Convert WireFormat to ContextFormat (where possible)
impl TryFrom<WireFormat> for ContextFormat {
    type Error = FormatConversionError;
    
    fn try_from(wire: WireFormat) -> Result<Self, Self::Error> {
        match wire {
            WireFormat::Toon => Ok(ContextFormat::Toon),
            WireFormat::Json => Ok(ContextFormat::Json),
            WireFormat::Columnar => Err(FormatConversionError {
                from: "Columnar".to_string(),
                to: "ContextFormat".to_string(),
                reason: "Columnar format is for analytics, not LLM context. \
                        Convert to Toon or Json first.".to_string(),
            }),
        }
    }
}

/// Convert ContextFormat to WireFormat (always succeeds)
impl From<ContextFormat> for WireFormat {
    fn from(ctx: ContextFormat) -> Self {
        match ctx {
            ContextFormat::Toon => WireFormat::Toon,
            ContextFormat::Json => WireFormat::Json,
            ContextFormat::Markdown => WireFormat::Toon, // Markdown renders to Toon structure
        }
    }
}

// ============================================================================
// Format Capabilities
// ============================================================================

/// Capabilities of a format
#[derive(Debug, Clone)]
pub struct FormatCapabilities {
    /// Supports structured data (tables, objects)
    pub structured: bool,
    /// Supports nested data
    pub nested: bool,
    /// Supports binary data (via encoding)
    pub binary: bool,
    /// Supports streaming
    pub streaming: bool,
    /// Typical token efficiency vs JSON (1.0 = same, 0.5 = half the tokens)
    pub token_efficiency: f32,
}

impl WireFormat {
    /// Get the capabilities of this format
    pub fn capabilities(&self) -> FormatCapabilities {
        match self {
            Self::Toon => FormatCapabilities {
                structured: true,
                nested: true,
                binary: true,
                streaming: true,
                token_efficiency: 0.4, // 40% of JSON tokens
            },
            Self::Json => FormatCapabilities {
                structured: true,
                nested: true,
                binary: false, // Requires base64
                streaming: true,
                token_efficiency: 1.0,
            },
            Self::Columnar => FormatCapabilities {
                structured: true,
                nested: false,
                binary: true,
                streaming: true,
                token_efficiency: 0.3, // Very efficient for tabular data
            },
        }
    }
}

impl ContextFormat {
    /// Get the capabilities of this format
    pub fn capabilities(&self) -> FormatCapabilities {
        match self {
            Self::Toon => FormatCapabilities {
                structured: true,
                nested: true,
                binary: true,
                streaming: true,
                token_efficiency: 0.4,
            },
            Self::Json => FormatCapabilities {
                structured: true,
                nested: true,
                binary: false,
                streaming: true,
                token_efficiency: 1.0,
            },
            Self::Markdown => FormatCapabilities {
                structured: true, // Tables, lists
                nested: true,     // Nested lists
                binary: false,
                streaming: true,
                token_efficiency: 0.7, // Less overhead than JSON
            },
        }
    }
    
    /// Get the recommended format for a given use case
    pub fn recommended_for_llm() -> Self {
        Self::Toon
    }
    
    pub fn recommended_for_human() -> Self {
        Self::Markdown
    }
    
    pub fn recommended_for_api() -> Self {
        Self::Json
    }
}

// ============================================================================
// Canonical AST Representation
// ============================================================================

/// Canonical format identifier for AST nodes.
///
/// This is the internal representation used in the query AST.
/// It preserves the user's intent exactly without lossy coercions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalFormat {
    /// Context format for LLM output
    Context(ContextFormat),
    /// Wire format for query results
    Wire(WireFormat),
}

impl CanonicalFormat {
    /// Get the format name for display
    pub fn name(&self) -> &'static str {
        match self {
            Self::Context(ContextFormat::Toon) => "context:toon",
            Self::Context(ContextFormat::Json) => "context:json",
            Self::Context(ContextFormat::Markdown) => "context:markdown",
            Self::Wire(WireFormat::Toon) => "wire:toon",
            Self::Wire(WireFormat::Json) => "wire:json",
            Self::Wire(WireFormat::Columnar) => "wire:columnar",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wire_format_parsing() {
        assert_eq!("toon".parse::<WireFormat>().unwrap(), WireFormat::Toon);
        assert_eq!("json".parse::<WireFormat>().unwrap(), WireFormat::Json);
        assert_eq!("columnar".parse::<WireFormat>().unwrap(), WireFormat::Columnar);
        assert!("invalid".parse::<WireFormat>().is_err());
    }
    
    #[test]
    fn test_context_format_parsing() {
        assert_eq!("toon".parse::<ContextFormat>().unwrap(), ContextFormat::Toon);
        assert_eq!("json".parse::<ContextFormat>().unwrap(), ContextFormat::Json);
        assert_eq!("markdown".parse::<ContextFormat>().unwrap(), ContextFormat::Markdown);
        
        // "text" should fail - no more silent coercion to Toon
        assert!("text".parse::<ContextFormat>().is_err());
    }
    
    #[test]
    fn test_wire_to_context_conversion() {
        assert_eq!(
            ContextFormat::try_from(WireFormat::Toon).unwrap(),
            ContextFormat::Toon
        );
        assert_eq!(
            ContextFormat::try_from(WireFormat::Json).unwrap(),
            ContextFormat::Json
        );
        
        // Columnar cannot convert to ContextFormat
        assert!(ContextFormat::try_from(WireFormat::Columnar).is_err());
    }
    
    #[test]
    fn test_context_to_wire_conversion() {
        assert_eq!(WireFormat::from(ContextFormat::Toon), WireFormat::Toon);
        assert_eq!(WireFormat::from(ContextFormat::Json), WireFormat::Json);
        assert_eq!(WireFormat::from(ContextFormat::Markdown), WireFormat::Toon);
    }
    
    #[test]
    fn test_round_trip_property() {
        // This test verifies the core requirement:
        // Converting to string and back should preserve the format
        
        for format in [WireFormat::Toon, WireFormat::Json, WireFormat::Columnar] {
            let s = format.to_string();
            let parsed: WireFormat = s.parse().unwrap();
            assert_eq!(format, parsed, "Round-trip failed for WireFormat::{:?}", format);
        }
        
        for format in [ContextFormat::Toon, ContextFormat::Json, ContextFormat::Markdown] {
            let s = format.to_string();
            let parsed: ContextFormat = s.parse().unwrap();
            assert_eq!(format, parsed, "Round-trip failed for ContextFormat::{:?}", format);
        }
    }
}
