# SochDB Analytics

SochDB includes optional, privacy-respecting analytics to help improve the database. This page explains what data is collected, how to disable analytics, and our privacy practices.

## What Is Collected

SochDB collects **anonymous usage metrics** to help us understand:

- Which SDK features are most used
- Performance characteristics (latency, throughput)
- Error patterns for debugging
- Platform distribution (OS, architecture)

### Example Event Data

```json
{
  "event": "vector_search",
  "properties": {
    "sdk": "python",
    "sdk_version": "0.3.1",
    "os": "Darwin",
    "arch": "arm64",
    "dimension": 1536,
    "k": 10,
    "latency_ms": 45.2
  },
  "distinct_id": "a1b2c3d4e5f6g7h8"  // Anonymous machine hash
}
```

## What Is NOT Collected

We **never** collect:

- ❌ Database contents or query data
- ❌ API keys or credentials
- ❌ Personal information (names, emails, IPs)
- ❌ File paths or directory structures
- ❌ Hostnames (only a hash is used for distinct_id)

## Disabling Analytics

To disable all analytics, set the environment variable:

```bash
# Bash/Zsh
export SOCHDB_DISABLE_ANALYTICS=true

# Windows PowerShell
$env:SOCHDB_DISABLE_ANALYTICS = "true"

# Windows CMD
set SOCHDB_DISABLE_ANALYTICS=true

# In Python
import os
os.environ["SOCHDB_DISABLE_ANALYTICS"] = "true"

# In Node.js
process.env.SOCHDB_DISABLE_ANALYTICS = "true";
```

### Verifying Analytics Status

#### Python
```python
from sochdb import is_analytics_disabled
print(f"Analytics disabled: {is_analytics_disabled()}")
```

#### JavaScript/TypeScript
```typescript
import { isAnalyticsDisabled } from '@sochdb/sochdb';
console.log(`Analytics disabled: ${isAnalyticsDisabled()}`);
```

#### Rust
```rust
use sochdb_core::analytics::is_analytics_disabled;
println!("Analytics disabled: {}", is_analytics_disabled());
```

## Analytics Provider

SochDB uses [PostHog](https://posthog.com) for analytics. PostHog is an open-source product analytics platform that respects user privacy and is GDPR compliant.

- **Data is sent to**: `https://us.i.posthog.com`
- **Data retention**: Aggregated metrics only
- **No third-party sharing**: Data is only used by SochDB developers

## Optional Dependency

The analytics package is **optional**:

- **Python**: Install with `pip install sochdb-client[analytics]`
- **Node.js**: posthog-node is in `optionalDependencies`
- **Rust**: Enable the `analytics` feature flag

If the analytics package is not installed, all tracking functions become no-ops.

## Events Tracked

| Event | Description | Properties |
|-------|-------------|------------|
| `database_opened` | Database connection established | mode, has_custom_path |
| `vector_search` | Vector similarity search performed | dimension, k, latency_ms |
| `batch_insert` | Batch vector insertion | count, dimension, latency_ms |
| `error` | Error occurred (sanitized) | error_type, error_message |

## Source Code

Analytics implementation is fully open source:

- Python: [sochdb-python-sdk/src/sochdb/analytics.py](https://github.com/sochdb/sochdb/blob/main/sochdb-python-sdk/src/sochdb/analytics.py)
- JavaScript: [sochdb-js/src/analytics.ts](https://github.com/sochdb/sochdb/blob/main/sochdb-js/src/analytics.ts)
- Rust: [sochdb-core/src/analytics.rs](https://github.com/sochdb/sochdb/blob/main/sochdb-core/src/analytics.rs)

## Questions?

If you have any questions or concerns about analytics, please [open an issue](https://github.com/sochdb/sochdb/issues) or email sushanth@sochdb.dev.
