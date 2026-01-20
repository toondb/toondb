# Multi-Agent Tool Routing

SochDB provides infrastructure for routing tool invocations across multiple
specialized agents. This enables building sophisticated multi-agent systems
with load balancing, capability-based routing, and persistent agent state.

## Overview

The routing system consists of three components:

| Component | Purpose |
|-----------|---------|
| `AgentRegistry` | Register agents and their tool capabilities |
| `ToolRouter` | Route tool invocations to appropriate agents |
| `ToolDispatcher` | High-level API combining registry + router |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tool Dispatcher                          │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐     ┌─────────────────────────────────┐ │
│ │ Agent Registry  │     │        Tool Router              │ │
│ │                 │     │                                 │ │
│ │ agent_001 ──────┼────►│ Strategy Selection:             │ │
│ │   - code_exec   │     │  • Round Robin                  │ │
│ │   - file_ops    │     │  • Least Loaded                 │ │
│ │                 │     │  • Priority                     │ │
│ │ agent_002 ──────┼────►│  • Sticky (session affinity)    │ │
│ │   - web_search  │     │  • Random                       │ │
│ │   - embeddings  │     │  • Fastest (latency-based)      │ │
│ └─────────────────┘     └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Python

```python
from sochdb import Database, ToolDispatcher, ToolCategory, RoutingStrategy

db = Database.open("./agent_data")
dispatcher = ToolDispatcher(db)

# Register agents with capabilities
dispatcher.register_agent(
    "code_agent",
    tools=[ToolCategory.CODE, ToolCategory.SHELL],
    endpoint="http://localhost:8001/invoke",
    priority=10
)

dispatcher.register_agent(
    "search_agent",
    tools=[ToolCategory.SEARCH, ToolCategory.WEB],
    endpoint="http://localhost:8002/invoke",
    priority=5
)

# Dispatch tool invocations
result = await dispatcher.invoke(
    tool="code_exec",
    category=ToolCategory.CODE,
    input={"code": "print('hello')"},
    strategy=RoutingStrategy.LEAST_LOADED
)
```

### Go

```go
db, _ := sochdb.Open("./agent_data")
dispatcher := sochdb.NewToolDispatcher(db)

// Register agents
dispatcher.RegisterAgent("code_agent", 
    sochdb.WithTools(sochdb.CategoryCode, sochdb.CategoryShell),
    sochdb.WithEndpoint("http://localhost:8001/invoke"),
    sochdb.WithPriority(10),
)

dispatcher.RegisterAgent("search_agent",
    sochdb.WithTools(sochdb.CategorySearch, sochdb.CategoryWeb),
    sochdb.WithEndpoint("http://localhost:8002/invoke"),
)

// Dispatch tool invocations
result, err := dispatcher.Invoke(context.Background(), "code_exec", sochdb.CategoryCode, 
    map[string]interface{}{"code": "print('hello')"},
    sochdb.WithStrategy(sochdb.StrategyLeastLoaded),
)
```

### TypeScript/Node.js

```typescript
import { Database, ToolDispatcher, ToolCategory, RoutingStrategy } from '@sochdb/sochdb';

const db = await Database.open('./agent_data');
const dispatcher = new ToolDispatcher(db);

// Register agents
await dispatcher.registerAgent('code_agent', {
  tools: [ToolCategory.CODE, ToolCategory.SHELL],
  endpoint: 'http://localhost:8001/invoke',
  priority: 10,
});

await dispatcher.registerAgent('search_agent', {
  tools: [ToolCategory.SEARCH, ToolCategory.WEB],
  endpoint: 'http://localhost:8002/invoke',
});

// Dispatch tool invocations
const result = await dispatcher.invoke('code_exec', ToolCategory.CODE, {
  code: "print('hello')",
}, {
  strategy: RoutingStrategy.LEAST_LOADED,
});
```

### Rust

```rust
use sochdb_client::routing::{ToolDispatcher, ToolCategory, RoutingStrategy, AgentConfig};

let conn = Connection::open("./agent_data")?;
let dispatcher = ToolDispatcher::new(conn);

// Register agents
dispatcher.register_agent(
    "code_agent",
    AgentConfig::new()
        .with_tools(vec![ToolCategory::Code, ToolCategory::Shell])
        .with_endpoint("http://localhost:8001/invoke")
        .with_priority(10)
)?;

// Dispatch tool invocations
let result = dispatcher.invoke(
    "code_exec",
    ToolCategory::Code,
    serde_json::json!({"code": "print('hello')"}),
    RoutingOptions::new().with_strategy(RoutingStrategy::LeastLoaded)
).await?;
```

## Tool Categories

Built-in tool categories for common agent capabilities:

| Category | Description | Example Tools |
|----------|-------------|---------------|
| `CODE` | Code execution | Python, Node.js interpreters |
| `SEARCH` | Search operations | Vector search, full-text search |
| `DATABASE` | Database operations | SQL queries, key-value ops |
| `WEB` | Web operations | HTTP requests, scraping |
| `FILE` | File system operations | Read, write, list files |
| `GIT` | Git operations | Clone, commit, push |
| `SHELL` | Shell commands | Bash, zsh execution |
| `EMBEDDING` | Vector embeddings | OpenAI, local models |

## Routing Strategies

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `ROUND_ROBIN` | Rotate through agents | Even distribution |
| `RANDOM` | Random selection | Simple load spreading |
| `LEAST_LOADED` | Pick agent with fewest active tasks | Optimal utilization |
| `PRIORITY` | Highest priority agent first | Prefer specialized agents |
| `STICKY` | Same agent for same session | Stateful conversations |
| `FASTEST` | Agent with lowest average latency | Latency-sensitive tasks |

### Sticky Sessions

Maintain session affinity for stateful agent interactions:

```python
# All invocations with same session_id go to same agent
result1 = await dispatcher.invoke(
    tool="chat",
    category=ToolCategory.CODE,
    input={"message": "Write a function"},
    session_id="session_abc123"
)

result2 = await dispatcher.invoke(
    tool="chat", 
    category=ToolCategory.CODE,
    input={"message": "Now test it"},
    session_id="session_abc123"  # Routes to same agent
)
```

## Agent State

Agent registrations are persisted to SochDB:

```python
# Registrations survive restarts
dispatcher.register_agent("agent_001", ...)

# Later, in different process
dispatcher = ToolDispatcher(db)
agents = dispatcher.list_agents()  # Returns ["agent_001", ...]
```

## Health Checks

Monitor agent health and automatically remove unhealthy agents:

```python
# Enable health checks
dispatcher.enable_health_checks(
    interval_seconds=30,
    timeout_seconds=5,
    unhealthy_threshold=3
)

# Agents are automatically marked unhealthy after 3 failed checks
# They're excluded from routing until health recovers
```

## Metrics

Track routing metrics for observability:

```python
metrics = dispatcher.get_metrics()

for agent_id, stats in metrics.items():
    print(f"{agent_id}:")
    print(f"  Total invocations: {stats['total']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Avg latency: {stats['avg_latency_ms']:.1f}ms")
    print(f"  Active tasks: {stats['active_tasks']}")
```

## Local Agent Handlers

For in-process agents, register handlers directly:

```python
# Register local handler function
async def code_executor(tool: str, input: dict) -> dict:
    result = exec(input["code"])
    return {"output": result}

dispatcher.register_local_agent(
    "local_code",
    handler=code_executor,
    tools=[ToolCategory.CODE]
)

# Invoke locally without HTTP
result = await dispatcher.invoke("code_exec", ToolCategory.CODE, {...})
```

## Combining with Policy Hooks

Use policies to enforce routing rules:

```python
from sochdb import PolicyEngine

policy = PolicyEngine(db)

# Require specific agents for sensitive tools
@policy.before_write("tools/sensitive/*")
def require_trusted_agent(key, value, context):
    if context.get("agent_id") not in ["trusted_agent_1", "trusted_agent_2"]:
        return PolicyAction.DENY
    return PolicyAction.ALLOW
```

## Best Practices

1. **Categorize Tools Properly**: Accurate categories enable better routing
2. **Use Sticky Sessions for Stateful Work**: Maintain context across invocations
3. **Set Realistic Priorities**: Higher priority = preferred for that category
4. **Enable Health Checks**: Auto-remove failing agents
5. **Monitor Metrics**: Track latency and success rates
6. **Prefer Local Handlers**: Use local handlers when agents are in-process

## See Also

- [Policy & Safety Hooks](policy-hooks.md) - Enforce security policies
- [Analytics Guide](analytics.md) - Track usage metrics
- [Deployment Guide](deployment.md) - Production deployment patterns
