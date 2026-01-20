# Policy & Safety Hooks

SochDB provides a first-class policy enforcement system for AI agent safety.
This enables pre-write validation, post-read filtering, rate limiting, and
audit logging to ensure agents operate within defined boundaries.

## Overview

The `PolicyEngine` wraps database operations with configurable hooks:

| Trigger | Purpose | Use Case |
|---------|---------|----------|
| `before_write` | Validate before writing | Block system key modifications |
| `after_write` | Post-write actions | Audit logging, notifications |
| `before_read` | Pre-read access control | Permission checks |
| `after_read` | Post-read filtering | Redact sensitive data |
| `before_delete` | Validate before delete | Protect critical data |

## Quick Start

### Python

```python
from sochdb import Database, PolicyEngine, PolicyAction

db = Database.open("./agent_data")
policy = PolicyEngine(db)

# Block writes to system keys from agents
@policy.before_write("system/*")
def block_system_writes(key, value, context):
    if context.get("agent_id"):
        return PolicyAction.DENY
    return PolicyAction.ALLOW

# Redact PII on read
@policy.after_read("users/*/email")
def redact_emails(key, value, context):
    if context.get("redact_pii"):
        return b"[REDACTED]"
    return value

# Use policy-wrapped operations
policy.put(b"users/alice", b"data", context={"agent_id": "agent_001"})
```

### Go

```go
db, _ := sochdb.Open("./agent_data")
policy := sochdb.NewPolicyEngine(db)

// Block writes to system keys
policy.BeforeWrite("system/*", func(ctx *sochdb.PolicyContext) sochdb.PolicyAction {
    if ctx.AgentID != "" {
        return sochdb.PolicyDeny
    }
    return sochdb.PolicyAllow
})

// Use policy-wrapped operations
err := policy.Put([]byte("users/alice"), []byte("data"), map[string]string{
    "agent_id": "agent_001",
})
```

### TypeScript/Node.js

```typescript
import { Database, PolicyEngine, PolicyAction } from '@sochdb/sochdb';

const db = await Database.open('./agent_data');
const policy = new PolicyEngine(db);

// Block writes to system keys
policy.beforeWrite('system/*', (ctx) => {
  if (ctx.agentId) {
    return PolicyAction.DENY;
  }
  return PolicyAction.ALLOW;
});

// Use policy-wrapped operations
await policy.put(Buffer.from('users/alice'), Buffer.from('data'), {
  agent_id: 'agent_001',
});
```

### Rust

```rust
use sochdb_client::policy::{PolicyEngine, PolicyAction, PolicyContext};

let conn = Connection::open("./agent_data")?;
let policy = PolicyEngine::new(conn);

// Block writes to system keys
policy.before_write("system/*", |ctx| {
    if ctx.agent_id.is_some() {
        PolicyAction::Deny
    } else {
        PolicyAction::Allow
    }
});

// Use policy-wrapped operations
let ctx = PolicyContext::new("write", b"users/alice")
    .with_agent_id("agent_001");
policy.put(b"users/alice", b"data", Some(&ctx))?;
```

## Pattern Matching

Patterns use glob-style matching:

| Pattern | Matches |
|---------|---------|
| `system/*` | `system/config`, `system/users` |
| `users/*/email` | `users/alice/email`, `users/bob/email` |
| `users/**` | `users/alice`, `users/alice/profile/photo` |
| `*.json` | `config.json`, `data.json` |

## Rate Limiting

Prevent runaway agents with token bucket rate limiting:

```python
# Global limit: 1000 writes per minute
policy.add_rate_limit("write", max_per_minute=1000, scope="global")

# Per-agent limit: 100 writes per minute per agent
policy.add_rate_limit("write", max_per_minute=100, scope="agent_id")

# Per-session limit
policy.add_rate_limit("read", max_per_minute=500, scope="session_id")
```

Scope options:
- `global` - Shared limit across all operations
- `agent_id` - Separate limit per agent
- `session_id` - Separate limit per session
- Any custom key in context

## Audit Logging

Track all agent operations for compliance and debugging:

```python
# Enable audit logging
policy.enable_audit(max_entries=10000)

# Perform operations...
policy.put(b"key", b"value", context={"agent_id": "agent_001"})

# Get recent audit entries
entries = policy.get_audit_log(limit=100)
for entry in entries:
    print(f"{entry['timestamp']}: {entry['operation']} {entry['key']} -> {entry['result']}")

# Filter by agent
agent_entries = policy.get_audit_log(limit=100, agent_id="agent_001")
```

## Built-in Policy Helpers

Common patterns are provided as helpers:

```python
from sochdb.policy import deny_all, allow_all, require_agent_id, redact_value

# Deny all operations matching pattern
policy.before_write("readonly/*", deny_all)

# Require agent_id in context
policy.before_write("agents/*", require_agent_id)

# Redact values on read
policy.after_read("secrets/*", redact_value(b"[REDACTED]"))
```

## Error Handling

When a policy blocks an operation, a `PolicyViolation` error is raised:

```python
from sochdb import PolicyViolation

try:
    policy.put(b"system/config", b"malicious", context={"agent_id": "rogue"})
except PolicyViolation as e:
    print(f"Operation blocked: {e}")
    # Log security event
```

## Best Practices

1. **Defense in Depth**: Combine multiple policies for layered security
2. **Audit Critical Operations**: Enable audit logging for sensitive namespaces
3. **Rate Limit by Agent**: Prevent any single agent from overwhelming the system
4. **Redact by Default**: Apply redaction policies to PII fields
5. **Test Policies**: Write unit tests for policy handlers
6. **Monitor Denials**: Alert on unusual denial patterns

## See Also

- [Tool Routing Guide](tool-routing.md) - Route tools to specialized agents
- [Multi-Tenancy Guide](../concepts/multi-tenancy.md) - Namespace isolation
- [Security Guide](../concepts/security.md) - Authentication and encryption
