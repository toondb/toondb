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

//! Unified Agent Execution Context (Task 16)
//!
//! Stateful session management for agentic use cases:
//! - Session variables and working directory
//! - Transaction scope (ACID across all operations)
//! - Permissions and sandboxing
//! - Audit logging for reproducibility
//!
//! ## Example
//!
//! ```text
//! Agent session abc123:
//!   cwd: /agents/abc123
//!   vars: $model = "gpt-4", $budget = 1000
//!   permissions: fs:rw, db:rw, calc:*
//!   audit: [read /data/users, write /agents/abc123/cache]
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Session identifier (UUID v7 for time-ordering)
pub type SessionId = String;

/// Agent session context
#[derive(Debug, Clone)]
pub struct AgentContext {
    /// Unique session identifier
    pub session_id: SessionId,
    /// Current working directory
    pub working_dir: String,
    /// Session variables
    pub variables: HashMap<String, ContextValue>,
    /// Permissions
    pub permissions: AgentPermissions,
    /// Session start time
    pub started_at: SystemTime,
    /// Last activity time
    pub last_activity: Instant,
    /// Audit trail
    pub audit: Vec<AuditEntry>,
    /// Transaction state
    pub transaction: Option<TransactionScope>,
    /// Maximum operation budget (token/cost limit)
    pub budget: OperationBudget,
    /// Tool registry: available tools for this agent
    pub tool_registry: Vec<ToolDefinition>,
    /// Tool call history for this session
    pub tool_calls: Vec<ToolCallRecord>,
}

/// Definition of a tool available to the agent
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Tool name/identifier
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// JSON Schema for parameters (if any)
    pub parameters_schema: Option<String>,
    /// Whether this tool requires confirmation
    pub requires_confirmation: bool,
}

/// Record of a tool call made during the session
#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    /// Unique call identifier
    pub call_id: String,
    /// Tool name
    pub tool_name: String,
    /// Arguments passed to the tool (JSON)
    pub arguments: String,
    /// Result if successful
    pub result: Option<String>,
    /// Error if failed
    pub error: Option<String>,
    /// Timestamp of the call
    pub timestamp: SystemTime,
}

/// Context variable value
#[derive(Debug, Clone, PartialEq)]
pub enum ContextValue {
    String(String),
    Number(f64),
    Bool(bool),
    List(Vec<ContextValue>),
    Object(HashMap<String, ContextValue>),
    Null,
}

impl fmt::Display for ContextValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextValue::String(s) => write!(f, "\"{}\"", s),
            ContextValue::Number(n) => write!(f, "{}", n),
            ContextValue::Bool(b) => write!(f, "{}", b),
            ContextValue::List(l) => {
                write!(f, "[")?;
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            ContextValue::Object(o) => {
                write!(f, "{{")?;
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{}\": {}", k, v)?;
                }
                write!(f, "}}")
            }
            ContextValue::Null => write!(f, "null"),
        }
    }
}

/// Agent permissions
#[derive(Debug, Clone, Default)]
pub struct AgentPermissions {
    /// Filesystem access
    pub filesystem: FsPermissions,
    /// Database access
    pub database: DbPermissions,
    /// Calculator access
    pub calculator: bool,
    /// Network access (future)
    pub network: NetworkPermissions,
}

/// Filesystem permissions
#[derive(Debug, Clone, Default)]
pub struct FsPermissions {
    /// Can read files
    pub read: bool,
    /// Can write files
    pub write: bool,
    /// Can create directories
    pub mkdir: bool,
    /// Can delete files/directories
    pub delete: bool,
    /// Allowed path prefixes (sandbox)
    pub allowed_paths: Vec<String>,
}

/// Database permissions
#[derive(Debug, Clone, Default)]
pub struct DbPermissions {
    /// Can read tables
    pub read: bool,
    /// Can write/insert
    pub write: bool,
    /// Can create tables
    pub create: bool,
    /// Can delete tables
    pub drop: bool,
    /// Allowed table patterns
    pub allowed_tables: Vec<String>,
}

/// Network permissions (for future use)
#[derive(Debug, Clone, Default)]
pub struct NetworkPermissions {
    /// Can make HTTP requests
    pub http: bool,
    /// Allowed domains
    pub allowed_domains: Vec<String>,
}

/// Audit trail entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Operation type
    pub operation: AuditOperation,
    /// Resource accessed
    pub resource: String,
    /// Result (success/error)
    pub result: AuditResult,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Audit operation types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditOperation {
    FsRead,
    FsWrite,
    FsMkdir,
    FsDelete,
    FsList,
    DbQuery,
    DbInsert,
    DbUpdate,
    DbDelete,
    Calculate,
    VarSet,
    VarGet,
    TxBegin,
    TxCommit,
    TxRollback,
}

impl fmt::Display for AuditOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuditOperation::FsRead => write!(f, "fs.read"),
            AuditOperation::FsWrite => write!(f, "fs.write"),
            AuditOperation::FsMkdir => write!(f, "fs.mkdir"),
            AuditOperation::FsDelete => write!(f, "fs.delete"),
            AuditOperation::FsList => write!(f, "fs.list"),
            AuditOperation::DbQuery => write!(f, "db.query"),
            AuditOperation::DbInsert => write!(f, "db.insert"),
            AuditOperation::DbUpdate => write!(f, "db.update"),
            AuditOperation::DbDelete => write!(f, "db.delete"),
            AuditOperation::Calculate => write!(f, "calc"),
            AuditOperation::VarSet => write!(f, "var.set"),
            AuditOperation::VarGet => write!(f, "var.get"),
            AuditOperation::TxBegin => write!(f, "tx.begin"),
            AuditOperation::TxCommit => write!(f, "tx.commit"),
            AuditOperation::TxRollback => write!(f, "tx.rollback"),
        }
    }
}

/// Audit result
#[derive(Debug, Clone)]
pub enum AuditResult {
    Success,
    Error(String),
    Denied(String),
}

/// Transaction scope
#[derive(Debug, Clone)]
pub struct TransactionScope {
    /// Transaction ID
    pub tx_id: u64,
    /// Started at
    pub started_at: Instant,
    /// Savepoints
    pub savepoints: Vec<String>,
    /// Pending writes (for rollback)
    pub pending_writes: Vec<PendingWrite>,
}

/// Pending write for transaction rollback
#[derive(Debug, Clone)]
pub struct PendingWrite {
    /// Resource type
    pub resource_type: ResourceType,
    /// Resource path/key
    pub resource_key: String,
    /// Original value (for rollback)
    pub original_value: Option<Vec<u8>>,
}

/// Resource type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceType {
    File,
    Directory,
    Table,
    Variable,
}

/// Operation budget tracking
#[derive(Debug, Clone)]
pub struct OperationBudget {
    /// Maximum tokens (input + output)
    pub max_tokens: Option<u64>,
    /// Tokens used
    pub tokens_used: u64,
    /// Maximum cost (in millicents)
    pub max_cost: Option<u64>,
    /// Cost used
    pub cost_used: u64,
    /// Maximum operations
    pub max_operations: Option<u64>,
    /// Operations used
    pub operations_used: u64,
}

impl Default for OperationBudget {
    fn default() -> Self {
        Self {
            max_tokens: None,
            max_cost: None,
            max_operations: Some(10000),
            tokens_used: 0,
            cost_used: 0,
            operations_used: 0,
        }
    }
}

/// Context error
#[derive(Debug, Clone)]
pub enum ContextError {
    PermissionDenied(String),
    VariableNotFound(String),
    BudgetExceeded(String),
    TransactionError(String),
    InvalidPath(String),
    SessionExpired,
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            ContextError::VariableNotFound(name) => write!(f, "Variable not found: {}", name),
            ContextError::BudgetExceeded(msg) => write!(f, "Budget exceeded: {}", msg),
            ContextError::TransactionError(msg) => write!(f, "Transaction error: {}", msg),
            ContextError::InvalidPath(path) => write!(f, "Invalid path: {}", path),
            ContextError::SessionExpired => write!(f, "Session expired"),
        }
    }
}

impl std::error::Error for ContextError {}

impl AgentContext {
    /// Create a new agent context
    pub fn new(session_id: SessionId) -> Self {
        let now = Instant::now();
        Self {
            session_id: session_id.clone(),
            working_dir: format!("/agents/{}", session_id),
            variables: HashMap::new(),
            permissions: AgentPermissions::default(),
            started_at: SystemTime::now(),
            last_activity: now,
            audit: Vec::new(),
            transaction: None,
            budget: OperationBudget::default(),
            tool_registry: Vec::new(),
            tool_calls: Vec::new(),
        }
    }

    /// Create with custom working directory
    pub fn with_working_dir(session_id: SessionId, working_dir: String) -> Self {
        let mut ctx = Self::new(session_id);
        ctx.working_dir = working_dir;
        ctx
    }

    /// Create with full permissions (for trusted agents)
    pub fn with_full_permissions(session_id: SessionId) -> Self {
        let mut ctx = Self::new(session_id);
        ctx.permissions = AgentPermissions {
            filesystem: FsPermissions {
                read: true,
                write: true,
                mkdir: true,
                delete: true,
                allowed_paths: vec!["/".into()],
            },
            database: DbPermissions {
                read: true,
                write: true,
                create: true,
                drop: true,
                allowed_tables: vec!["*".into()],
            },
            calculator: true,
            network: NetworkPermissions::default(),
        };
        ctx
    }

    /// Register a tool with this agent
    pub fn register_tool(&mut self, tool: ToolDefinition) {
        self.tool_registry.push(tool);
    }

    /// Record a tool call
    pub fn record_tool_call(&mut self, call: ToolCallRecord) {
        self.tool_calls.push(call);
    }

    /// Set a variable
    pub fn set_var(&mut self, name: &str, value: ContextValue) {
        self.variables.insert(name.to_string(), value.clone());
        self.touch();
        self.audit(AuditOperation::VarSet, name, AuditResult::Success);
    }

    /// Get a variable (returns cloned value to avoid borrow issues)
    pub fn get_var(&mut self, name: &str) -> Option<ContextValue> {
        self.touch();
        let result = self.variables.get(name).cloned();
        if result.is_some() {
            self.audit(AuditOperation::VarGet, name, AuditResult::Success);
        } else {
            self.audit(
                AuditOperation::VarGet,
                name,
                AuditResult::Error("not found".into()),
            );
        }
        result
    }

    /// Get a variable reference without auditing (for read-only access)
    pub fn peek_var(&self, name: &str) -> Option<&ContextValue> {
        self.variables.get(name)
    }

    /// Update last activity time
    fn touch(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Add audit entry
    fn audit(&mut self, operation: AuditOperation, resource: &str, result: AuditResult) {
        self.audit.push(AuditEntry {
            timestamp: SystemTime::now(),
            operation,
            resource: resource.to_string(),
            result,
            metadata: HashMap::new(),
        });
    }

    /// Check filesystem permission
    pub fn check_fs_permission(&self, path: &str, op: AuditOperation) -> Result<(), ContextError> {
        let perm = match op {
            AuditOperation::FsRead | AuditOperation::FsList => self.permissions.filesystem.read,
            AuditOperation::FsWrite => self.permissions.filesystem.write,
            AuditOperation::FsMkdir => self.permissions.filesystem.mkdir,
            AuditOperation::FsDelete => self.permissions.filesystem.delete,
            _ => {
                return Err(ContextError::PermissionDenied(
                    "invalid fs operation".into(),
                ));
            }
        };

        if !perm {
            return Err(ContextError::PermissionDenied(format!(
                "{} not allowed",
                op
            )));
        }

        // Check path sandbox
        if !self.permissions.filesystem.allowed_paths.is_empty() {
            let allowed = self
                .permissions
                .filesystem
                .allowed_paths
                .iter()
                .any(|p| path.starts_with(p) || p == "*");
            if !allowed {
                return Err(ContextError::PermissionDenied(format!(
                    "path {} not in allowed paths",
                    path
                )));
            }
        }

        Ok(())
    }

    /// Check database permission
    pub fn check_db_permission(&self, table: &str, op: AuditOperation) -> Result<(), ContextError> {
        let perm = match op {
            AuditOperation::DbQuery => self.permissions.database.read,
            AuditOperation::DbInsert | AuditOperation::DbUpdate => self.permissions.database.write,
            AuditOperation::DbDelete => self.permissions.database.drop,
            _ => {
                return Err(ContextError::PermissionDenied(
                    "invalid db operation".into(),
                ));
            }
        };

        if !perm {
            return Err(ContextError::PermissionDenied(format!(
                "{} not allowed",
                op
            )));
        }

        // Check table pattern
        if !self.permissions.database.allowed_tables.is_empty() {
            let allowed = self.permissions.database.allowed_tables.iter().any(|t| {
                t == "*" || t == table || (t.ends_with('*') && table.starts_with(&t[..t.len() - 1]))
            });
            if !allowed {
                return Err(ContextError::PermissionDenied(format!(
                    "table {} not in allowed tables",
                    table
                )));
            }
        }

        Ok(())
    }

    /// Consume operation budget
    pub fn consume_budget(&mut self, tokens: u64, cost: u64) -> Result<(), ContextError> {
        self.budget.operations_used += 1;
        self.budget.tokens_used += tokens;
        self.budget.cost_used += cost;

        if let Some(max) = self.budget.max_operations
            && self.budget.operations_used > max
        {
            return Err(ContextError::BudgetExceeded("max operations".into()));
        }
        if let Some(max) = self.budget.max_tokens
            && self.budget.tokens_used > max
        {
            return Err(ContextError::BudgetExceeded("max tokens".into()));
        }
        if let Some(max) = self.budget.max_cost
            && self.budget.cost_used > max
        {
            return Err(ContextError::BudgetExceeded("max cost".into()));
        }

        Ok(())
    }

    /// Begin transaction
    pub fn begin_transaction(&mut self, tx_id: u64) -> Result<(), ContextError> {
        if self.transaction.is_some() {
            return Err(ContextError::TransactionError(
                "already in transaction".into(),
            ));
        }

        self.transaction = Some(TransactionScope {
            tx_id,
            started_at: Instant::now(),
            savepoints: Vec::new(),
            pending_writes: Vec::new(),
        });

        self.audit(
            AuditOperation::TxBegin,
            &format!("tx:{}", tx_id),
            AuditResult::Success,
        );
        Ok(())
    }

    /// Commit transaction
    pub fn commit_transaction(&mut self) -> Result<(), ContextError> {
        let tx = self
            .transaction
            .take()
            .ok_or_else(|| ContextError::TransactionError("no active transaction".into()))?;

        self.audit(
            AuditOperation::TxCommit,
            &format!("tx:{}", tx.tx_id),
            AuditResult::Success,
        );
        Ok(())
    }

    /// Rollback transaction
    pub fn rollback_transaction(&mut self) -> Result<Vec<PendingWrite>, ContextError> {
        let tx = self
            .transaction
            .take()
            .ok_or_else(|| ContextError::TransactionError("no active transaction".into()))?;

        self.audit(
            AuditOperation::TxRollback,
            &format!("tx:{}", tx.tx_id),
            AuditResult::Success,
        );

        Ok(tx.pending_writes)
    }

    /// Add savepoint
    pub fn savepoint(&mut self, name: &str) -> Result<(), ContextError> {
        let tx = self
            .transaction
            .as_mut()
            .ok_or_else(|| ContextError::TransactionError("no active transaction".into()))?;

        tx.savepoints.push(name.to_string());
        Ok(())
    }

    /// Record pending write for rollback
    pub fn record_pending_write(
        &mut self,
        resource_type: ResourceType,
        resource_key: String,
        original_value: Option<Vec<u8>>,
    ) -> Result<(), ContextError> {
        let tx = self
            .transaction
            .as_mut()
            .ok_or_else(|| ContextError::TransactionError("no active transaction".into()))?;

        tx.pending_writes.push(PendingWrite {
            resource_type,
            resource_key,
            original_value,
        });
        Ok(())
    }

    /// Resolve path relative to working directory
    pub fn resolve_path(&self, path: &str) -> String {
        if path.starts_with('/') {
            path.to_string()
        } else {
            format!("{}/{}", self.working_dir, path)
        }
    }

    /// Substitute variables in string ($var syntax)
    pub fn substitute_vars(&self, input: &str) -> String {
        let mut result = input.to_string();

        for (name, value) in &self.variables {
            let pattern = format!("${}", name);
            let replacement = match value {
                ContextValue::String(s) => s.clone(),
                ContextValue::Number(n) => n.to_string(),
                ContextValue::Bool(b) => b.to_string(),
                _ => value.to_string(),
            };
            result = result.replace(&pattern, &replacement);
        }

        result
    }

    /// Get session age
    pub fn age(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.started_at)
            .unwrap_or_default()
    }

    /// Get idle time
    pub fn idle_time(&self) -> Duration {
        self.last_activity.elapsed()
    }

    /// Check if session is expired (default: 1 hour idle)
    pub fn is_expired(&self, idle_timeout: Duration) -> bool {
        self.idle_time() > idle_timeout
    }

    /// Export audit log
    pub fn export_audit(&self) -> Vec<HashMap<String, String>> {
        self.audit
            .iter()
            .map(|entry| {
                let mut m = HashMap::new();
                m.insert(
                    "timestamp".into(),
                    entry
                        .timestamp
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_secs().to_string())
                        .unwrap_or_default(),
                );
                m.insert("operation".into(), entry.operation.to_string());
                m.insert("resource".into(), entry.resource.clone());
                m.insert(
                    "result".into(),
                    match &entry.result {
                        AuditResult::Success => "success".into(),
                        AuditResult::Error(e) => format!("error:{}", e),
                        AuditResult::Denied(r) => format!("denied:{}", r),
                    },
                );
                for (k, v) in &entry.metadata {
                    m.insert(k.clone(), v.clone());
                }
                m
            })
            .collect()
    }
}

/// Session manager for multiple agent contexts
pub struct SessionManager {
    sessions: RwLock<HashMap<SessionId, Arc<RwLock<AgentContext>>>>,
    idle_timeout: Duration,
}

impl SessionManager {
    /// Create new session manager
    pub fn new(idle_timeout: Duration) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            idle_timeout,
        }
    }

    /// Create a new session
    pub fn create_session(&self, session_id: SessionId) -> Arc<RwLock<AgentContext>> {
        let ctx = Arc::new(RwLock::new(AgentContext::new(session_id.clone())));
        self.sessions
            .write()
            .unwrap()
            .insert(session_id, ctx.clone());
        ctx
    }

    /// Get existing session
    pub fn get_session(&self, session_id: &str) -> Option<Arc<RwLock<AgentContext>>> {
        let sessions = self.sessions.read().unwrap();
        sessions.get(session_id).cloned()
    }

    /// Get or create session
    pub fn get_or_create(&self, session_id: SessionId) -> Arc<RwLock<AgentContext>> {
        if let Some(ctx) = self.get_session(&session_id) {
            return ctx;
        }
        self.create_session(session_id)
    }

    /// Remove session
    pub fn remove_session(&self, session_id: &str) -> Option<Arc<RwLock<AgentContext>>> {
        self.sessions.write().unwrap().remove(session_id)
    }

    /// Cleanup expired sessions
    pub fn cleanup_expired(&self) -> usize {
        let mut sessions = self.sessions.write().unwrap();
        let initial_count = sessions.len();

        sessions.retain(|_, ctx| {
            let ctx = ctx.read().unwrap();
            !ctx.is_expired(self.idle_timeout)
        });

        initial_count - sessions.len()
    }

    /// Get active session count
    pub fn session_count(&self) -> usize {
        self.sessions.read().unwrap().len()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new(Duration::from_secs(3600)) // 1 hour
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = AgentContext::new("test-session".into());
        assert_eq!(ctx.session_id, "test-session");
        assert_eq!(ctx.working_dir, "/agents/test-session");
    }

    #[test]
    fn test_variables() {
        let mut ctx = AgentContext::new("test".into());
        ctx.set_var("model", ContextValue::String("gpt-4".into()));
        ctx.set_var("budget", ContextValue::Number(1000.0));

        assert_eq!(
            ctx.get_var("model"),
            Some(ContextValue::String("gpt-4".into()))
        );
        assert_eq!(ctx.get_var("budget"), Some(ContextValue::Number(1000.0)));
    }

    #[test]
    fn test_variable_substitution() {
        let mut ctx = AgentContext::new("test".into());
        ctx.set_var("name", ContextValue::String("Alice".into()));
        ctx.set_var("count", ContextValue::Number(42.0));

        let result = ctx.substitute_vars("Hello $name, you have $count items");
        assert_eq!(result, "Hello Alice, you have 42 items");
    }

    #[test]
    fn test_path_resolution() {
        let ctx = AgentContext::with_working_dir("test".into(), "/home/agent".into());

        assert_eq!(ctx.resolve_path("data.json"), "/home/agent/data.json");
        assert_eq!(ctx.resolve_path("/absolute/path"), "/absolute/path");
    }

    #[test]
    fn test_permissions() {
        let mut ctx = AgentContext::new("test".into());
        ctx.permissions.filesystem.read = true;
        ctx.permissions.filesystem.allowed_paths = vec!["/allowed".into()];

        assert!(
            ctx.check_fs_permission("/allowed/file", AuditOperation::FsRead)
                .is_ok()
        );
        assert!(
            ctx.check_fs_permission("/forbidden/file", AuditOperation::FsRead)
                .is_err()
        );
        assert!(
            ctx.check_fs_permission("/allowed/file", AuditOperation::FsWrite)
                .is_err()
        );
    }

    #[test]
    fn test_budget() {
        let mut ctx = AgentContext::new("test".into());
        ctx.budget.max_operations = Some(3);

        assert!(ctx.consume_budget(100, 10).is_ok());
        assert!(ctx.consume_budget(100, 10).is_ok());
        assert!(ctx.consume_budget(100, 10).is_ok());
        assert!(ctx.consume_budget(100, 10).is_err());
    }

    #[test]
    fn test_transaction() {
        let mut ctx = AgentContext::new("test".into());

        assert!(ctx.begin_transaction(1).is_ok());
        assert!(ctx.begin_transaction(2).is_err()); // Already in tx

        ctx.record_pending_write(
            ResourceType::File,
            "/test/file".into(),
            Some(b"original".to_vec()),
        )
        .unwrap();

        let pending = ctx.rollback_transaction().unwrap();
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_session_manager() {
        let mgr = SessionManager::default();

        let _s1 = mgr.create_session("s1".into());
        let _s2 = mgr.create_session("s2".into());

        assert_eq!(mgr.session_count(), 2);
        assert!(mgr.get_session("s1").is_some());
        assert!(mgr.get_session("s3").is_none());

        mgr.remove_session("s1");
        assert_eq!(mgr.session_count(), 1);
    }
}
