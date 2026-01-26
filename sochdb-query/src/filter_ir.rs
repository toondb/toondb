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

//! Canonical Filter IR + Planner Pushdown Contract (Task 1)
//!
//! This module defines a **single source of truth** for filtering behavior across
//! all retrieval paths (vector/BM25/hybrid/context). By normalizing filters to a
//! canonical IR and enforcing pushdown contracts, we:
//!
//! 1. **Prevent post-filtering by construction** - filters are applied during
//!    candidate generation, not after
//! 2. **Unify semantics** - "namespace = X" means the same thing everywhere
//! 3. **Enable systematic optimization** - CNF form allows index path selection
//!
//! ## Filter IR Design
//!
//! Filters are normalized to **Conjunctive Normal Form (CNF)**: a conjunction
//! of disjunctions of typed atoms.
//!
//! ```text
//! EffectiveFilter = AuthScope ∧ UserFilter
//!                 = (A₁ ∨ A₂) ∧ (B₁) ∧ (C₁ ∨ C₂ ∨ C₃)
//! ```
//!
//! Where each atom is a typed predicate:
//! - `Eq(field, value)` - equality
//! - `In(field, values)` - membership in set
//! - `Range(field, min, max)` - inclusive range
//! - `HasTag(tag)` - ACL tag presence (future)
//!
//! ## Pushdown Contract
//!
//! Every executor MUST implement:
//! ```text
//! execute(query_op, filter_ir, auth_scope) -> results
//! ```
//!
//! The executor guarantees:
//! 1. All returned results satisfy `filter_ir ∧ auth_scope`
//! 2. No result outside the allowed set is ever generated
//! 3. Filter application happens BEFORE scoring (no post-filter)
//!
//! ## Auth Scope
//!
//! `AuthScope` is **non-optional** and always conjoined with user filters:
//! ```text
//! EffectiveFilter = AuthScope ∧ UserFilter
//! ```
//!
//! This is a monotone strengthening (can only remove results, never add),
//! ensuring security invariants hold.

use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

// ============================================================================
// Filter Atoms - Typed Predicates
// ============================================================================

/// A typed scalar value for filter comparison
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterValue {
    /// String value
    String(String),
    /// 64-bit signed integer
    Int64(i64),
    /// 64-bit unsigned integer (for doc_id, timestamps)
    Uint64(u64),
    /// 64-bit float
    Float64(f64),
    /// Boolean
    Bool(bool),
    /// Null
    Null,
}

impl FilterValue {
    /// Check if this value matches another for equality
    pub fn eq_match(&self, other: &FilterValue) -> bool {
        match (self, other) {
            (FilterValue::String(a), FilterValue::String(b)) => a == b,
            (FilterValue::Int64(a), FilterValue::Int64(b)) => a == b,
            (FilterValue::Uint64(a), FilterValue::Uint64(b)) => a == b,
            (FilterValue::Float64(a), FilterValue::Float64(b)) => {
                (a - b).abs() < f64::EPSILON
            }
            (FilterValue::Bool(a), FilterValue::Bool(b)) => a == b,
            (FilterValue::Null, FilterValue::Null) => true,
            _ => false,
        }
    }

    /// Compare for ordering (returns None if incompatible types)
    pub fn partial_cmp(&self, other: &FilterValue) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (FilterValue::Int64(a), FilterValue::Int64(b)) => Some(a.cmp(b)),
            (FilterValue::Uint64(a), FilterValue::Uint64(b)) => Some(a.cmp(b)),
            (FilterValue::Float64(a), FilterValue::Float64(b)) => a.partial_cmp(b),
            (FilterValue::String(a), FilterValue::String(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }
}

impl fmt::Display for FilterValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterValue::String(s) => write!(f, "'{}'", s),
            FilterValue::Int64(i) => write!(f, "{}", i),
            FilterValue::Uint64(u) => write!(f, "{}u64", u),
            FilterValue::Float64(v) => write!(f, "{}", v),
            FilterValue::Bool(b) => write!(f, "{}", b),
            FilterValue::Null => write!(f, "NULL"),
        }
    }
}

impl From<&str> for FilterValue {
    fn from(s: &str) -> Self {
        FilterValue::String(s.to_string())
    }
}

impl From<String> for FilterValue {
    fn from(s: String) -> Self {
        FilterValue::String(s)
    }
}

impl From<i64> for FilterValue {
    fn from(i: i64) -> Self {
        FilterValue::Int64(i)
    }
}

impl From<u64> for FilterValue {
    fn from(u: u64) -> Self {
        FilterValue::Uint64(u)
    }
}

impl From<f64> for FilterValue {
    fn from(f: f64) -> Self {
        FilterValue::Float64(f)
    }
}

impl From<bool> for FilterValue {
    fn from(b: bool) -> Self {
        FilterValue::Bool(b)
    }
}

/// A single filter atom - the smallest unit of filtering
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterAtom {
    /// Equality: field = value
    Eq {
        field: String,
        value: FilterValue,
    },
    
    /// Not equal: field != value
    Ne {
        field: String,
        value: FilterValue,
    },
    
    /// Membership: field IN (v1, v2, ...)
    In {
        field: String,
        values: Vec<FilterValue>,
    },
    
    /// Not in set: field NOT IN (v1, v2, ...)
    NotIn {
        field: String,
        values: Vec<FilterValue>,
    },
    
    /// Range: min <= field <= max (inclusive)
    /// Either bound can be None for open-ended ranges
    Range {
        field: String,
        min: Option<FilterValue>,
        max: Option<FilterValue>,
        min_inclusive: bool,
        max_inclusive: bool,
    },
    
    /// Prefix match: field STARTS WITH prefix
    Prefix {
        field: String,
        prefix: String,
    },
    
    /// Contains substring: field CONTAINS substring
    Contains {
        field: String,
        substring: String,
    },
    
    /// ACL tag presence (for row-level security)
    HasTag {
        tag: String,
    },
    
    /// Always true (identity for conjunction)
    True,
    
    /// Always false (identity for disjunction)
    False,
}

impl FilterAtom {
    /// Create an equality atom
    pub fn eq(field: impl Into<String>, value: impl Into<FilterValue>) -> Self {
        FilterAtom::Eq {
            field: field.into(),
            value: value.into(),
        }
    }
    
    /// Create an IN atom
    pub fn in_set(field: impl Into<String>, values: Vec<FilterValue>) -> Self {
        FilterAtom::In {
            field: field.into(),
            values,
        }
    }
    
    /// Create a range atom
    pub fn range(
        field: impl Into<String>,
        min: Option<FilterValue>,
        max: Option<FilterValue>,
    ) -> Self {
        FilterAtom::Range {
            field: field.into(),
            min,
            max,
            min_inclusive: true,
            max_inclusive: true,
        }
    }
    
    /// Create an open range (exclusive bounds)
    pub fn range_exclusive(
        field: impl Into<String>,
        min: Option<FilterValue>,
        max: Option<FilterValue>,
    ) -> Self {
        FilterAtom::Range {
            field: field.into(),
            min,
            max,
            min_inclusive: false,
            max_inclusive: false,
        }
    }
    
    /// Get the field name this atom filters on (if any)
    pub fn field(&self) -> Option<&str> {
        match self {
            FilterAtom::Eq { field, .. } => Some(field),
            FilterAtom::Ne { field, .. } => Some(field),
            FilterAtom::In { field, .. } => Some(field),
            FilterAtom::NotIn { field, .. } => Some(field),
            FilterAtom::Range { field, .. } => Some(field),
            FilterAtom::Prefix { field, .. } => Some(field),
            FilterAtom::Contains { field, .. } => Some(field),
            FilterAtom::HasTag { .. } => None,
            FilterAtom::True | FilterAtom::False => None,
        }
    }
    
    /// Check if this atom is always true
    pub fn is_trivially_true(&self) -> bool {
        matches!(self, FilterAtom::True)
    }
    
    /// Check if this atom is always false
    pub fn is_trivially_false(&self) -> bool {
        matches!(self, FilterAtom::False)
    }
    
    /// Negate this atom
    pub fn negate(&self) -> FilterAtom {
        match self {
            FilterAtom::Eq { field, value } => FilterAtom::Ne {
                field: field.clone(),
                value: value.clone(),
            },
            FilterAtom::Ne { field, value } => FilterAtom::Eq {
                field: field.clone(),
                value: value.clone(),
            },
            FilterAtom::In { field, values } => FilterAtom::NotIn {
                field: field.clone(),
                values: values.clone(),
            },
            FilterAtom::NotIn { field, values } => FilterAtom::In {
                field: field.clone(),
                values: values.clone(),
            },
            FilterAtom::True => FilterAtom::False,
            FilterAtom::False => FilterAtom::True,
            // For complex atoms, wrap in negation via De Morgan's
            other => other.clone(), // Simplified - full implementation would use Not wrapper
        }
    }
}

impl fmt::Display for FilterAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterAtom::Eq { field, value } => write!(f, "{} = {}", field, value),
            FilterAtom::Ne { field, value } => write!(f, "{} != {}", field, value),
            FilterAtom::In { field, values } => {
                let vals: Vec<_> = values.iter().map(|v| v.to_string()).collect();
                write!(f, "{} IN ({})", field, vals.join(", "))
            }
            FilterAtom::NotIn { field, values } => {
                let vals: Vec<_> = values.iter().map(|v| v.to_string()).collect();
                write!(f, "{} NOT IN ({})", field, vals.join(", "))
            }
            FilterAtom::Range { field, min, max, min_inclusive, max_inclusive } => {
                let left = if *min_inclusive { "[" } else { "(" };
                let right = if *max_inclusive { "]" } else { ")" };
                let min_str = min.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "-∞".to_string());
                let max_str = max.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "∞".to_string());
                write!(f, "{} ∈ {}{}, {}{}", field, left, min_str, max_str, right)
            }
            FilterAtom::Prefix { field, prefix } => write!(f, "{} STARTS WITH '{}'", field, prefix),
            FilterAtom::Contains { field, substring } => write!(f, "{} CONTAINS '{}'", field, substring),
            FilterAtom::HasTag { tag } => write!(f, "HAS_TAG('{}')", tag),
            FilterAtom::True => write!(f, "TRUE"),
            FilterAtom::False => write!(f, "FALSE"),
        }
    }
}

// ============================================================================
// Filter IR - Normalized Boolean Expression
// ============================================================================

/// A disjunction (OR) of atoms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Disjunction {
    pub atoms: Vec<FilterAtom>,
}

impl Disjunction {
    /// Create a disjunction from atoms
    pub fn new(atoms: Vec<FilterAtom>) -> Self {
        Self { atoms }
    }
    
    /// Create a single-atom disjunction
    pub fn single(atom: FilterAtom) -> Self {
        Self { atoms: vec![atom] }
    }
    
    /// Check if this disjunction is trivially true (contains TRUE or is empty after simplification)
    pub fn is_trivially_true(&self) -> bool {
        self.atoms.iter().any(|a| a.is_trivially_true())
    }
    
    /// Check if this disjunction is trivially false (empty or all atoms are FALSE)
    pub fn is_trivially_false(&self) -> bool {
        self.atoms.is_empty() || self.atoms.iter().all(|a| a.is_trivially_false())
    }
    
    /// Simplify this disjunction
    pub fn simplify(self) -> Self {
        // Remove FALSE atoms
        let atoms: Vec<_> = self.atoms.into_iter()
            .filter(|a| !a.is_trivially_false())
            .collect();
        
        // If any atom is TRUE, the whole disjunction is TRUE
        if atoms.iter().any(|a| a.is_trivially_true()) {
            return Self { atoms: vec![FilterAtom::True] };
        }
        
        // If empty, it's FALSE
        if atoms.is_empty() {
            return Self { atoms: vec![FilterAtom::False] };
        }
        
        Self { atoms }
    }
}

impl fmt::Display for Disjunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.atoms.len() == 1 {
            write!(f, "{}", self.atoms[0])
        } else {
            let parts: Vec<_> = self.atoms.iter().map(|a| a.to_string()).collect();
            write!(f, "({})", parts.join(" OR "))
        }
    }
}

/// Canonical Filter IR in Conjunctive Normal Form (CNF)
///
/// CNF = (A₁ ∨ A₂) ∧ (B₁) ∧ (C₁ ∨ C₂ ∨ C₃)
///
/// This representation enables:
/// 1. Systematic index path selection (each clause maps to an index)
/// 2. Easy conjunction with auth scope (just append clauses)
/// 3. Efficient serialization and transmission
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterIR {
    /// Conjunction of disjunctions (CNF form)
    pub clauses: Vec<Disjunction>,
}

impl FilterIR {
    /// Create an empty filter (matches everything)
    pub fn all() -> Self {
        Self { clauses: vec![] }
    }
    
    /// Create a filter that matches nothing
    pub fn none() -> Self {
        Self {
            clauses: vec![Disjunction::single(FilterAtom::False)],
        }
    }
    
    /// Create a filter from a single atom
    pub fn from_atom(atom: FilterAtom) -> Self {
        Self {
            clauses: vec![Disjunction::single(atom)],
        }
    }
    
    /// Create a filter from a single disjunction
    pub fn from_disjunction(disj: Disjunction) -> Self {
        Self { clauses: vec![disj] }
    }
    
    /// Conjoin (AND) with another filter
    ///
    /// This is the key operation for auth scope injection:
    /// `EffectiveFilter = AuthScope ∧ UserFilter`
    pub fn and(mut self, other: FilterIR) -> Self {
        self.clauses.extend(other.clauses);
        self
    }
    
    /// Conjoin with a single atom
    pub fn and_atom(mut self, atom: FilterAtom) -> Self {
        self.clauses.push(Disjunction::single(atom));
        self
    }
    
    /// Disjoin (OR) with another filter
    ///
    /// Note: This may expand the CNF representation
    pub fn or(self, other: FilterIR) -> Self {
        if self.clauses.is_empty() {
            return other;
        }
        if other.clauses.is_empty() {
            return self;
        }
        
        // Distribute: (A ∧ B) ∨ (C ∧ D) = (A ∨ C) ∧ (A ∨ D) ∧ (B ∨ C) ∧ (B ∨ D)
        // This can cause exponential blowup - in practice, limit depth
        let mut new_clauses = Vec::new();
        for c1 in &self.clauses {
            for c2 in &other.clauses {
                let mut combined = c1.atoms.clone();
                combined.extend(c2.atoms.clone());
                new_clauses.push(Disjunction::new(combined));
            }
        }
        
        FilterIR { clauses: new_clauses }
    }
    
    /// Check if this filter matches everything
    pub fn is_all(&self) -> bool {
        self.clauses.is_empty() || self.clauses.iter().all(|c| c.is_trivially_true())
    }
    
    /// Check if this filter matches nothing
    pub fn is_none(&self) -> bool {
        self.clauses.iter().any(|c| c.is_trivially_false())
    }
    
    /// Simplify the filter
    pub fn simplify(self) -> Self {
        let clauses: Vec<_> = self.clauses
            .into_iter()
            .map(|c| c.simplify())
            .filter(|c| !c.is_trivially_true())
            .collect();
        
        // If any clause is FALSE, the whole conjunction is FALSE
        if clauses.iter().any(|c| c.is_trivially_false()) {
            return Self::none();
        }
        
        Self { clauses }
    }
    
    /// Extract atoms for a specific field
    pub fn atoms_for_field(&self, field: &str) -> Vec<&FilterAtom> {
        self.clauses
            .iter()
            .flat_map(|c| c.atoms.iter())
            .filter(|a| a.field() == Some(field))
            .collect()
    }
    
    /// Check if this filter constrains a specific field
    pub fn constrains_field(&self, field: &str) -> bool {
        !self.atoms_for_field(field).is_empty()
    }
    
    /// Get all fields constrained by this filter
    pub fn constrained_fields(&self) -> HashSet<&str> {
        self.clauses
            .iter()
            .flat_map(|c| c.atoms.iter())
            .filter_map(|a| a.field())
            .collect()
    }
}

impl Default for FilterIR {
    fn default() -> Self {
        Self::all()
    }
}

impl fmt::Display for FilterIR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.clauses.is_empty() {
            return write!(f, "TRUE");
        }
        let parts: Vec<_> = self.clauses.iter().map(|c| c.to_string()).collect();
        write!(f, "{}", parts.join(" AND "))
    }
}

// ============================================================================
// Auth Scope - Non-Optional Security Context
// ============================================================================

/// Authorization scope - ALWAYS conjoined with user filters
///
/// This is the security boundary that cannot be bypassed. It encodes:
/// - Allowed namespaces/tenants
/// - Optional project scope
/// - Token expiry
/// - Capability flags
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthScope {
    /// Allowed namespaces (non-empty; at least one required)
    pub allowed_namespaces: Vec<String>,
    
    /// Optional tenant ID (for multi-tenant deployments)
    pub tenant_id: Option<String>,
    
    /// Optional project scope
    pub project_id: Option<String>,
    
    /// Token expiry timestamp (Unix epoch seconds)
    pub expires_at: Option<u64>,
    
    /// Capability flags
    pub capabilities: AuthCapabilities,
    
    /// Optional ACL tags the caller has access to
    pub acl_tags: Vec<String>,
}

/// Capability flags for authorization
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AuthCapabilities {
    /// Can read vectors
    pub can_read: bool,
    /// Can write/insert vectors
    pub can_write: bool,
    /// Can delete vectors
    pub can_delete: bool,
    /// Can perform admin operations
    pub can_admin: bool,
}

impl AuthScope {
    /// Create a new auth scope for a single namespace
    pub fn for_namespace(namespace: impl Into<String>) -> Self {
        Self {
            allowed_namespaces: vec![namespace.into()],
            tenant_id: None,
            project_id: None,
            expires_at: None,
            capabilities: AuthCapabilities {
                can_read: true,
                can_write: false,
                can_delete: false,
                can_admin: false,
            },
            acl_tags: vec![],
        }
    }
    
    /// Create with full access to a namespace
    pub fn full_access(namespace: impl Into<String>) -> Self {
        Self {
            allowed_namespaces: vec![namespace.into()],
            tenant_id: None,
            project_id: None,
            expires_at: None,
            capabilities: AuthCapabilities {
                can_read: true,
                can_write: true,
                can_delete: true,
                can_admin: false,
            },
            acl_tags: vec![],
        }
    }
    
    /// Add a namespace to the allowed list
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.allowed_namespaces.push(namespace.into());
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
    
    /// Set expiry
    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }
    
    /// Add ACL tags
    pub fn with_acl_tags(mut self, tags: Vec<String>) -> Self {
        self.acl_tags = tags;
        self
    }
    
    /// Check if this scope is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            now > expires_at
        } else {
            false
        }
    }
    
    /// Check if a namespace is allowed
    pub fn is_namespace_allowed(&self, namespace: &str) -> bool {
        self.allowed_namespaces.iter().any(|ns| ns == namespace)
    }
    
    /// Convert auth scope to filter IR clauses
    ///
    /// This generates the mandatory predicates that MUST be conjoined
    /// with any user filter.
    pub fn to_filter_ir(&self) -> FilterIR {
        let mut filter = FilterIR::all();
        
        // Namespace constraint (mandatory)
        if self.allowed_namespaces.len() == 1 {
            filter = filter.and_atom(FilterAtom::eq(
                "namespace",
                self.allowed_namespaces[0].clone(),
            ));
        } else if !self.allowed_namespaces.is_empty() {
            filter = filter.and_atom(FilterAtom::in_set(
                "namespace",
                self.allowed_namespaces
                    .iter()
                    .map(|ns| FilterValue::String(ns.clone()))
                    .collect(),
            ));
        }
        
        // Tenant constraint (if present)
        if let Some(ref tenant_id) = self.tenant_id {
            filter = filter.and_atom(FilterAtom::eq("tenant_id", tenant_id.clone()));
        }
        
        // Project constraint (if present)
        if let Some(ref project_id) = self.project_id {
            filter = filter.and_atom(FilterAtom::eq("project_id", project_id.clone()));
        }
        
        // ACL tags (if present, user must have at least one matching tag)
        // This is handled differently - the executor checks tag intersection
        // rather than adding to filter IR (since it's "has any of these tags")
        
        filter
    }
}

// ============================================================================
// Pushdown Contract - Executor Interface
// ============================================================================

/// The pushdown contract that every executor MUST implement
///
/// This trait enforces:
/// 1. Filter is provided upfront, not as post-processing
/// 2. Auth scope is non-optional
/// 3. Results are guaranteed to satisfy the effective filter
pub trait FilteredExecutor {
    /// The query operation type (varies by executor)
    type QueryOp;
    
    /// The result type
    type Result;
    
    /// The error type
    type Error;
    
    /// Execute a query with mandatory filtering
    ///
    /// # Contract
    ///
    /// - `filter_ir`: User-provided filter (may be empty = all)
    /// - `auth_scope`: Non-optional security context
    ///
    /// The executor MUST:
    /// 1. Compute `effective_filter = auth_scope.to_filter_ir() ∧ filter_ir`
    /// 2. Apply `effective_filter` BEFORE generating candidates
    /// 3. Guarantee all results satisfy `effective_filter`
    ///
    /// The executor MUST NOT:
    /// 1. Return any result outside `effective_filter`
    /// 2. Apply filtering after candidate scoring
    /// 3. Ignore or bypass `auth_scope`
    fn execute(
        &self,
        query: &Self::QueryOp,
        filter_ir: &FilterIR,
        auth_scope: &AuthScope,
    ) -> Result<Self::Result, Self::Error>;
    
    /// Compute the effective filter (auth ∧ user)
    ///
    /// This is a convenience method that executors can use.
    fn effective_filter(&self, filter_ir: &FilterIR, auth_scope: &AuthScope) -> FilterIR {
        auth_scope.to_filter_ir().and(filter_ir.clone())
    }
}

// ============================================================================
// Filter Builder - Ergonomic Construction
// ============================================================================

/// Builder for constructing filter IR ergonomically
#[derive(Debug, Clone, Default)]
pub struct FilterBuilder {
    clauses: Vec<Disjunction>,
}

impl FilterBuilder {
    /// Create a new filter builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add an equality constraint
    pub fn eq(mut self, field: &str, value: impl Into<FilterValue>) -> Self {
        self.clauses.push(Disjunction::single(FilterAtom::eq(field, value)));
        self
    }
    
    /// Add a not-equal constraint
    pub fn ne(mut self, field: &str, value: impl Into<FilterValue>) -> Self {
        self.clauses.push(Disjunction::single(FilterAtom::Ne {
            field: field.to_string(),
            value: value.into(),
        }));
        self
    }
    
    /// Add an IN constraint
    pub fn in_set(mut self, field: &str, values: Vec<FilterValue>) -> Self {
        self.clauses.push(Disjunction::single(FilterAtom::in_set(field, values)));
        self
    }
    
    /// Add a range constraint
    pub fn range(
        mut self,
        field: &str,
        min: Option<impl Into<FilterValue>>,
        max: Option<impl Into<FilterValue>>,
    ) -> Self {
        self.clauses.push(Disjunction::single(FilterAtom::range(
            field,
            min.map(Into::into),
            max.map(Into::into),
        )));
        self
    }
    
    /// Add a greater-than constraint
    pub fn gt(mut self, field: &str, value: impl Into<FilterValue>) -> Self {
        self.clauses.push(Disjunction::single(FilterAtom::Range {
            field: field.to_string(),
            min: Some(value.into()),
            max: None,
            min_inclusive: false,
            max_inclusive: false,
        }));
        self
    }
    
    /// Add a greater-than-or-equal constraint
    pub fn gte(mut self, field: &str, value: impl Into<FilterValue>) -> Self {
        self.clauses.push(Disjunction::single(FilterAtom::Range {
            field: field.to_string(),
            min: Some(value.into()),
            max: None,
            min_inclusive: true,
            max_inclusive: false,
        }));
        self
    }
    
    /// Add a less-than constraint
    pub fn lt(mut self, field: &str, value: impl Into<FilterValue>) -> Self {
        self.clauses.push(Disjunction::single(FilterAtom::Range {
            field: field.to_string(),
            min: None,
            max: Some(value.into()),
            min_inclusive: false,
            max_inclusive: false,
        }));
        self
    }
    
    /// Add a less-than-or-equal constraint
    pub fn lte(mut self, field: &str, value: impl Into<FilterValue>) -> Self {
        self.clauses.push(Disjunction::single(FilterAtom::Range {
            field: field.to_string(),
            min: None,
            max: Some(value.into()),
            min_inclusive: false,
            max_inclusive: true,
        }));
        self
    }
    
    /// Add a prefix match constraint
    pub fn prefix(mut self, field: &str, prefix: &str) -> Self {
        self.clauses.push(Disjunction::single(FilterAtom::Prefix {
            field: field.to_string(),
            prefix: prefix.to_string(),
        }));
        self
    }
    
    /// Add a contains constraint
    pub fn contains(mut self, field: &str, substring: &str) -> Self {
        self.clauses.push(Disjunction::single(FilterAtom::Contains {
            field: field.to_string(),
            substring: substring.to_string(),
        }));
        self
    }
    
    /// Add a namespace constraint (convenience method)
    pub fn namespace(self, namespace: &str) -> Self {
        self.eq("namespace", namespace)
    }
    
    /// Add a doc_id IN constraint (convenience method)
    pub fn doc_ids(self, doc_ids: &[u64]) -> Self {
        self.in_set(
            "doc_id",
            doc_ids.iter().map(|&id| FilterValue::Uint64(id)).collect(),
        )
    }
    
    /// Add a time range constraint (convenience method)
    pub fn time_range(self, field: &str, start: Option<u64>, end: Option<u64>) -> Self {
        self.range(
            field,
            start.map(FilterValue::Uint64),
            end.map(FilterValue::Uint64),
        )
    }
    
    /// Add a disjunction (OR of multiple atoms)
    pub fn or_atoms(mut self, atoms: Vec<FilterAtom>) -> Self {
        self.clauses.push(Disjunction::new(atoms));
        self
    }
    
    /// Build the filter IR
    pub fn build(self) -> FilterIR {
        FilterIR { clauses: self.clauses }
    }
}

// ============================================================================
// Convenience Macros
// ============================================================================

/// Create a filter IR from a simple DSL
///
/// ```ignore
/// let filter = filter_ir! {
///     namespace = "my_ns",
///     project_id = "proj_123",
///     timestamp in 1000..2000
/// };
/// ```
#[macro_export]
macro_rules! filter_ir {
    // Empty filter
    () => {
        $crate::filter_ir::FilterIR::all()
    };
    
    // Equality
    ($field:ident = $value:expr $(, $($rest:tt)*)?) => {{
        let mut builder = $crate::filter_ir::FilterBuilder::new()
            .eq(stringify!($field), $value);
        $(
            builder = filter_ir!(@chain builder, $($rest)*);
        )?
        builder.build()
    }};
    
    // Chaining helper
    (@chain $builder:expr, $field:ident = $value:expr $(, $($rest:tt)*)?) => {{
        let builder = $builder.eq(stringify!($field), $value);
        $(
            filter_ir!(@chain builder, $($rest)*)
        )?
        builder
    }};
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_filter_atom_creation() {
        let eq = FilterAtom::eq("namespace", "my_ns");
        assert_eq!(eq.field(), Some("namespace"));
        
        let range = FilterAtom::range("timestamp", Some(FilterValue::Uint64(1000)), Some(FilterValue::Uint64(2000)));
        assert_eq!(range.field(), Some("timestamp"));
    }
    
    #[test]
    fn test_filter_ir_conjunction() {
        let filter1 = FilterIR::from_atom(FilterAtom::eq("namespace", "ns1"));
        let filter2 = FilterIR::from_atom(FilterAtom::eq("project_id", "proj1"));
        
        let combined = filter1.and(filter2);
        assert_eq!(combined.clauses.len(), 2);
    }
    
    #[test]
    fn test_auth_scope_to_filter() {
        let scope = AuthScope::for_namespace("production")
            .with_tenant("acme_corp");
        
        let filter = scope.to_filter_ir();
        assert!(filter.constrains_field("namespace"));
        assert!(filter.constrains_field("tenant_id"));
        assert!(!filter.constrains_field("project_id"));
    }
    
    #[test]
    fn test_effective_filter() {
        let auth = AuthScope::for_namespace("production");
        let user_filter = FilterBuilder::new()
            .eq("source", "documents")
            .time_range("created_at", Some(1000), Some(2000))
            .build();
        
        let effective = auth.to_filter_ir().and(user_filter);
        
        // Should have namespace + source + time range
        assert_eq!(effective.clauses.len(), 3);
        assert!(effective.constrains_field("namespace"));
        assert!(effective.constrains_field("source"));
        assert!(effective.constrains_field("created_at"));
    }
    
    #[test]
    fn test_filter_builder() {
        let filter = FilterBuilder::new()
            .namespace("my_namespace")
            .eq("project_id", "proj_123")
            .doc_ids(&[1, 2, 3, 4, 5])
            .time_range("timestamp", Some(1000), None)
            .build();
        
        assert_eq!(filter.clauses.len(), 4);
    }
    
    #[test]
    fn test_filter_simplification() {
        // TRUE AND X = X
        let filter = FilterIR::from_atom(FilterAtom::True)
            .and(FilterIR::from_atom(FilterAtom::eq("x", "y")));
        let simplified = filter.simplify();
        assert_eq!(simplified.clauses.len(), 1);
        
        // FALSE AND X = FALSE
        let filter2 = FilterIR::from_atom(FilterAtom::False)
            .and(FilterIR::from_atom(FilterAtom::eq("x", "y")));
        let simplified2 = filter2.simplify();
        assert!(simplified2.is_none());
    }
    
    #[test]
    fn test_filter_display() {
        let filter = FilterBuilder::new()
            .eq("namespace", "prod")
            .range("timestamp", Some(1000i64), Some(2000i64))
            .build();
        
        let display = filter.to_string();
        assert!(display.contains("namespace"));
        assert!(display.contains("timestamp"));
    }
}
