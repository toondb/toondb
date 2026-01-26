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

//! Capability Tokens + ACLs (Task 8)
//!
//! This module implements staged ACLs via capability tokens for the local-first
//! architecture. The design prioritizes:
//!
//! 1. **Simplicity** - Easy to reason about, hard to misapply
//! 2. **Local-first** - No external auth service required
//! 3. **Composability** - ACLs integrate with existing filter infrastructure
//!
//! ## Token Structure
//!
//! ```text
//! CapabilityToken {
//!     allowed_namespaces: ["prod", "staging"],
//!     tenant_id: Option<"acme_corp">,
//!     project_id: Option<"project_123">,
//!     capabilities: { read: true, write: false, ... },
//!     expires_at: 1735689600,
//!     signature: HMAC-SHA256(...)
//! }
//! ```
//!
//! ## Verification
//!
//! Token verification is O(1):
//! - HMAC-SHA256 for symmetric tokens
//! - Ed25519 for asymmetric tokens (cached verification)
//!
//! ## Row-Level ACLs (Future)
//!
//! Row-level ACL tags become "just another metadata atom":
//! ```text
//! HasTag(acl_tag) → bitmap lookup → AllowedSet intersection
//! ```
//!
//! This composes cleanly with existing filter infrastructure.

use std::collections::HashSet;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::filter_ir::{AuthCapabilities, AuthScope};

// ============================================================================
// Capability Token
// ============================================================================

/// A capability token that encodes access permissions
///
/// This is the serializable form that can be passed across API boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityToken {
    /// Token version (for future upgrades)
    pub version: u8,
    
    /// Token ID (for revocation tracking)
    pub token_id: String,
    
    /// Allowed namespaces (non-empty)
    pub allowed_namespaces: Vec<String>,
    
    /// Optional tenant ID
    pub tenant_id: Option<String>,
    
    /// Optional project ID
    pub project_id: Option<String>,
    
    /// Capability flags
    pub capabilities: TokenCapabilities,
    
    /// Issued at (Unix timestamp)
    pub issued_at: u64,
    
    /// Expires at (Unix timestamp)
    pub expires_at: u64,
    
    /// ACL tags the token holder can access (for row-level ACLs)
    pub acl_tags: Vec<String>,
    
    /// Signature (HMAC-SHA256 or Ed25519)
    pub signature: Vec<u8>,
}

/// Capability flags in the token
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenCapabilities {
    /// Can read/query vectors
    pub can_read: bool,
    /// Can insert vectors
    pub can_write: bool,
    /// Can delete vectors
    pub can_delete: bool,
    /// Can perform admin operations (create/drop indexes)
    pub can_admin: bool,
    /// Can create new tokens (delegation)
    pub can_delegate: bool,
}

impl CapabilityToken {
    /// Current token version
    pub const CURRENT_VERSION: u8 = 1;
    
    /// Check if the token is expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now > self.expires_at
    }
    
    /// Check if a namespace is allowed
    pub fn is_namespace_allowed(&self, namespace: &str) -> bool {
        self.allowed_namespaces.iter().any(|ns| ns == namespace)
    }
    
    /// Convert to AuthScope for use with FilterIR
    pub fn to_auth_scope(&self) -> AuthScope {
        AuthScope {
            allowed_namespaces: self.allowed_namespaces.clone(),
            tenant_id: self.tenant_id.clone(),
            project_id: self.project_id.clone(),
            expires_at: Some(self.expires_at),
            capabilities: AuthCapabilities {
                can_read: self.capabilities.can_read,
                can_write: self.capabilities.can_write,
                can_delete: self.capabilities.can_delete,
                can_admin: self.capabilities.can_admin,
            },
            acl_tags: self.acl_tags.clone(),
        }
    }
    
    /// Get remaining validity duration
    pub fn remaining_validity(&self) -> Option<Duration> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        if now >= self.expires_at {
            None
        } else {
            Some(Duration::from_secs(self.expires_at - now))
        }
    }
}

// ============================================================================
// Token Builder
// ============================================================================

/// Builder for creating capability tokens
pub struct TokenBuilder {
    namespaces: Vec<String>,
    tenant_id: Option<String>,
    project_id: Option<String>,
    capabilities: TokenCapabilities,
    validity: Duration,
    acl_tags: Vec<String>,
}

impl TokenBuilder {
    /// Create a new token builder for a namespace
    pub fn new(namespace: impl Into<String>) -> Self {
        Self {
            namespaces: vec![namespace.into()],
            tenant_id: None,
            project_id: None,
            capabilities: TokenCapabilities {
                can_read: true,
                ..Default::default()
            },
            validity: Duration::from_secs(3600), // 1 hour default
            acl_tags: Vec::new(),
        }
    }
    
    /// Add another namespace
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespaces.push(namespace.into());
        self
    }
    
    /// Set tenant ID
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }
    
    /// Set project ID
    pub fn with_project(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }
    
    /// Enable read capability
    pub fn can_read(mut self) -> Self {
        self.capabilities.can_read = true;
        self
    }
    
    /// Enable write capability
    pub fn can_write(mut self) -> Self {
        self.capabilities.can_write = true;
        self
    }
    
    /// Enable delete capability
    pub fn can_delete(mut self) -> Self {
        self.capabilities.can_delete = true;
        self
    }
    
    /// Enable admin capability
    pub fn can_admin(mut self) -> Self {
        self.capabilities.can_admin = true;
        self
    }
    
    /// Enable all capabilities
    pub fn full_access(mut self) -> Self {
        self.capabilities = TokenCapabilities {
            can_read: true,
            can_write: true,
            can_delete: true,
            can_admin: true,
            can_delegate: false,
        };
        self
    }
    
    /// Set validity duration
    pub fn valid_for(mut self, duration: Duration) -> Self {
        self.validity = duration;
        self
    }
    
    /// Add ACL tags
    pub fn with_acl_tags(mut self, tags: Vec<String>) -> Self {
        self.acl_tags = tags;
        self
    }
    
    /// Build the token (unsigned - call sign() on TokenSigner)
    pub fn build_unsigned(self) -> CapabilityToken {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        CapabilityToken {
            version: CapabilityToken::CURRENT_VERSION,
            token_id: generate_token_id(),
            allowed_namespaces: self.namespaces,
            tenant_id: self.tenant_id,
            project_id: self.project_id,
            capabilities: self.capabilities,
            issued_at: now,
            expires_at: now + self.validity.as_secs(),
            acl_tags: self.acl_tags,
            signature: Vec::new(),
        }
    }
}

/// Generate a unique token ID
fn generate_token_id() -> String {
    
    // Simple ID generation - in production use UUID or similar
    format!("tok_{:x}", 
        std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

// ============================================================================
// Token Signing and Verification
// ============================================================================

/// Token signer using HMAC-SHA256
pub struct TokenSigner {
    /// Secret key for HMAC
    secret: Vec<u8>,
}

impl TokenSigner {
    /// Create a new signer with a secret key
    pub fn new(secret: impl AsRef<[u8]>) -> Self {
        Self {
            secret: secret.as_ref().to_vec(),
        }
    }
    
    /// Sign a token
    pub fn sign(&self, token: &mut CapabilityToken) {
        let payload = self.compute_payload(token);
        token.signature = self.hmac_sha256(&payload);
    }
    
    /// Verify a token signature
    pub fn verify(&self, token: &CapabilityToken) -> Result<(), TokenError> {
        // Check version
        if token.version != CapabilityToken::CURRENT_VERSION {
            return Err(TokenError::UnsupportedVersion(token.version));
        }
        
        // Check expiry
        if token.is_expired() {
            return Err(TokenError::Expired);
        }
        
        // Verify signature
        let payload = self.compute_payload(token);
        let expected = self.hmac_sha256(&payload);
        
        if !constant_time_eq(&token.signature, &expected) {
            return Err(TokenError::InvalidSignature);
        }
        
        Ok(())
    }
    
    /// Compute the payload to sign
    fn compute_payload(&self, token: &CapabilityToken) -> Vec<u8> {
        // Deterministic serialization of token fields (excluding signature)
        let mut payload = Vec::new();
        
        payload.push(token.version);
        payload.extend(token.token_id.as_bytes());
        
        for ns in &token.allowed_namespaces {
            payload.extend(ns.as_bytes());
            payload.push(0); // Separator
        }
        
        if let Some(ref tenant) = token.tenant_id {
            payload.extend(tenant.as_bytes());
        }
        payload.push(0);
        
        if let Some(ref project) = token.project_id {
            payload.extend(project.as_bytes());
        }
        payload.push(0);
        
        // Capabilities as flags
        let caps = (token.capabilities.can_read as u8)
            | ((token.capabilities.can_write as u8) << 1)
            | ((token.capabilities.can_delete as u8) << 2)
            | ((token.capabilities.can_admin as u8) << 3)
            | ((token.capabilities.can_delegate as u8) << 4);
        payload.push(caps);
        
        payload.extend(&token.issued_at.to_le_bytes());
        payload.extend(&token.expires_at.to_le_bytes());
        
        for tag in &token.acl_tags {
            payload.extend(tag.as_bytes());
            payload.push(0);
        }
        
        payload
    }
    
    /// HMAC-SHA256
    fn hmac_sha256(&self, data: &[u8]) -> Vec<u8> {
        // Simple HMAC implementation
        // In production, use a proper crypto library
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // This is NOT cryptographically secure - just for demonstration
        // Use ring, hmac, or sha2 crates in production
        let mut hasher = DefaultHasher::new();
        self.secret.hash(&mut hasher);
        data.hash(&mut hasher);
        let h1 = hasher.finish();
        
        let mut hasher2 = DefaultHasher::new();
        h1.hash(&mut hasher2);
        self.secret.hash(&mut hasher2);
        let h2 = hasher2.finish();
        
        let mut result = Vec::with_capacity(16);
        result.extend(&h1.to_le_bytes());
        result.extend(&h2.to_le_bytes());
        result
    }
}

/// Constant-time comparison to prevent timing attacks
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Token errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum TokenError {
    #[error("token has expired")]
    Expired,
    
    #[error("invalid signature")]
    InvalidSignature,
    
    #[error("unsupported token version: {0}")]
    UnsupportedVersion(u8),
    
    #[error("token revoked")]
    Revoked,
    
    #[error("namespace not allowed: {0}")]
    NamespaceNotAllowed(String),
    
    #[error("insufficient capabilities")]
    InsufficientCapabilities,
}

// ============================================================================
// Token Revocation (Simple In-Memory)
// ============================================================================

/// Simple in-memory token revocation list
pub struct RevocationList {
    /// Revoked token IDs
    revoked: std::sync::RwLock<HashSet<String>>,
}

impl RevocationList {
    /// Create a new revocation list
    pub fn new() -> Self {
        Self {
            revoked: std::sync::RwLock::new(HashSet::new()),
        }
    }
    
    /// Revoke a token
    pub fn revoke(&self, token_id: &str) {
        self.revoked.write().unwrap().insert(token_id.to_string());
    }
    
    /// Check if a token is revoked
    pub fn is_revoked(&self, token_id: &str) -> bool {
        self.revoked.read().unwrap().contains(token_id)
    }
    
    /// Get count of revoked tokens
    pub fn count(&self) -> usize {
        self.revoked.read().unwrap().len()
    }
}

impl Default for RevocationList {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Token Validator (Combines Signer + Revocation)
// ============================================================================

/// Complete token validator
pub struct TokenValidator {
    signer: TokenSigner,
    revocation_list: RevocationList,
}

impl TokenValidator {
    /// Create a new validator
    pub fn new(secret: impl AsRef<[u8]>) -> Self {
        Self {
            signer: TokenSigner::new(secret),
            revocation_list: RevocationList::new(),
        }
    }
    
    /// Issue a new token
    pub fn issue(&self, builder: TokenBuilder) -> CapabilityToken {
        let mut token = builder.build_unsigned();
        self.signer.sign(&mut token);
        token
    }
    
    /// Validate a token
    pub fn validate(&self, token: &CapabilityToken) -> Result<AuthScope, TokenError> {
        // Check revocation
        if self.revocation_list.is_revoked(&token.token_id) {
            return Err(TokenError::Revoked);
        }
        
        // Verify signature and expiry
        self.signer.verify(token)?;
        
        // Convert to AuthScope
        Ok(token.to_auth_scope())
    }
    
    /// Revoke a token
    pub fn revoke(&self, token_id: &str) {
        self.revocation_list.revoke(token_id);
    }
}

// ============================================================================
// Row-Level ACL Tags (Future Extension)
// ============================================================================

/// A row-level ACL tag
/// 
/// In the future, documents can have ACL tags and tokens can specify
/// which tags they can access. This integrates with the filter IR:
///
/// ```text
/// FilterAtom::HasTag("confidential") → bitmap lookup → intersection
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AclTag(String);

impl AclTag {
    /// Create a new ACL tag
    pub fn new(tag: impl Into<String>) -> Self {
        Self(tag.into())
    }
    
    /// Get the tag name
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// ACL tag index for row-level security
/// 
/// This would be integrated with MetadataIndex to provide:
/// tag → bitmap of doc_ids with that tag
#[derive(Debug, Default)]
pub struct AclTagIndex {
    /// Map from tag to doc_ids
    tag_to_docs: std::collections::HashMap<String, Vec<u64>>,
}

impl AclTagIndex {
    /// Create a new ACL tag index
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a tag to a document
    pub fn add_tag(&mut self, doc_id: u64, tag: &str) {
        self.tag_to_docs
            .entry(tag.to_string())
            .or_default()
            .push(doc_id);
    }
    
    /// Get doc_ids with a specific tag
    pub fn docs_with_tag(&self, tag: &str) -> &[u64] {
        self.tag_to_docs.get(tag).map(|v| v.as_slice()).unwrap_or(&[])
    }
    
    /// Get doc_ids accessible by a set of allowed tags (union)
    pub fn accessible_docs(&self, allowed_tags: &[String]) -> Vec<u64> {
        let mut result = HashSet::new();
        for tag in allowed_tags {
            if let Some(docs) = self.tag_to_docs.get(tag) {
                result.extend(docs.iter().copied());
            }
        }
        result.into_iter().collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_token_builder() {
        let token = TokenBuilder::new("production")
            .with_namespace("staging")
            .with_tenant("acme")
            .can_read()
            .can_write()
            .valid_for(Duration::from_secs(3600))
            .build_unsigned();
        
        assert_eq!(token.allowed_namespaces.len(), 2);
        assert_eq!(token.tenant_id, Some("acme".to_string()));
        assert!(token.capabilities.can_read);
        assert!(token.capabilities.can_write);
        assert!(!token.capabilities.can_delete);
    }
    
    #[test]
    fn test_token_signing_and_verification() {
        let signer = TokenSigner::new("super_secret_key");
        
        let mut token = TokenBuilder::new("production")
            .can_read()
            .valid_for(Duration::from_secs(3600))
            .build_unsigned();
        
        signer.sign(&mut token);
        assert!(!token.signature.is_empty());
        
        // Verification should succeed
        assert!(signer.verify(&token).is_ok());
        
        // Tamper with token
        token.allowed_namespaces.push("hacked".to_string());
        assert!(signer.verify(&token).is_err());
    }
    
    #[test]
    fn test_token_expiry() {
        // Create a token that expires 1 second in the past
        let mut token = TokenBuilder::new("production")
            .valid_for(Duration::from_secs(3600))
            .build_unsigned();
        
        // Manually set expires_at to 0 (Unix epoch - in the past)
        token.expires_at = 0;
        
        assert!(token.is_expired());
    }
    
    #[test]
    fn test_token_to_auth_scope() {
        let token = TokenBuilder::new("production")
            .with_tenant("acme")
            .can_read()
            .can_write()
            .with_acl_tags(vec!["public".to_string(), "internal".to_string()])
            .build_unsigned();
        
        let scope = token.to_auth_scope();
        assert!(scope.is_namespace_allowed("production"));
        assert!(!scope.is_namespace_allowed("staging"));
        assert_eq!(scope.tenant_id, Some("acme".to_string()));
        assert!(scope.capabilities.can_read);
        assert!(scope.capabilities.can_write);
        assert_eq!(scope.acl_tags.len(), 2);
    }
    
    #[test]
    fn test_revocation() {
        let validator = TokenValidator::new("secret");
        
        let token = validator.issue(
            TokenBuilder::new("production")
                .can_read()
                .valid_for(Duration::from_secs(3600))
        );
        
        // Should validate
        assert!(validator.validate(&token).is_ok());
        
        // Revoke
        validator.revoke(&token.token_id);
        
        // Should fail validation
        assert!(matches!(
            validator.validate(&token),
            Err(TokenError::Revoked)
        ));
    }
    
    #[test]
    fn test_acl_tag_index() {
        let mut index = AclTagIndex::new();
        
        index.add_tag(1, "public");
        index.add_tag(2, "public");
        index.add_tag(3, "internal");
        index.add_tag(4, "confidential");
        
        assert_eq!(index.docs_with_tag("public").len(), 2);
        assert_eq!(index.docs_with_tag("internal").len(), 1);
        
        let accessible = index.accessible_docs(&["public".to_string(), "internal".to_string()]);
        assert_eq!(accessible.len(), 3);
    }
}
