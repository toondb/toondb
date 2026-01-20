# SochDB v0.3.1 Release Notes

**Release Date:** January 4, 2026  
**Type:** Minor Release  
**Status:** Stable

---

## 🎯 Overview

SochDB v0.3.1 introduces **optional, privacy-respecting usage analytics** to help improve the database while maintaining user privacy. This release adds anonymous usage information collection with full transparency and user control.

---

## ✨ What's New

### Anonymous Usage Analytics

Optional telemetry to understand how SochDB is used in production, with strict privacy guarantees:

#### Events Tracked

1. **`database_opened`** — Sent once when database is initialized
   - Properties: `mode` (embedded/server), `has_custom_path` (boolean)
   - Helps understand deployment patterns

2. **`error`** — Static error tracking for reliability improvements
   - Properties: `error_type` (category), `location` (code path)
   - **No dynamic error messages** — only static identifiers
   - Example: `"query_error"` at `"sql.execute"` (no SQL queries, no user data)

#### Privacy Guarantees

✅ **Anonymous ID** — Stable SHA-256 hash of machine info (hostname, OS, arch)  
✅ **No PII** — No usernames, file paths, query content, or error messages  
✅ **Opt-out** — Set `SOCHDB_DISABLE_ANALYTICS=true` to disable completely  
✅ **Optional Dependencies** — Graceful degradation if analytics libraries unavailable  
✅ **Open Source** — All analytics code visible in repository

#### SDK Implementation

**Python SDK:**
```python
from sochdb import Database
from sochdb.analytics import capture_error

# Analytics automatically tracks database_opened
db = Database.open("./mydb")

# Track errors (static info only)
try:
    db.execute(query)
except Exception:
    capture_error("query_error", "sql.execute")
```

Installation with analytics:
```bash
pip install sochdb[analytics]  # Includes posthog
# or
pip install sochdb  # Works without analytics
```

**JavaScript SDK:**
```javascript
import { Database, captureError } from '@sochdb/sochdb';

// Analytics automatically tracks database_opened
const db = await Database.open('./mydb');

// Track errors (static info only)
try {
  await db.execute(query);
} catch (err) {
  await captureError('query_error', 'sql.execute');
}
```

Installation:
```bash
npm install @sochdb/sochdb
# posthog-node is optionalDependency (works without it)
```

**Rust SDK:**
```rust
use sochdb_core::analytics;

// Enable analytics feature in Cargo.toml
// sochdb-core = { version = "0.3.1", features = ["analytics"] }

let db = Database::open("./mydb")?;
analytics::analytics().track_database_open("./mydb", "embedded");

// Track errors
if let Err(_) = db.query(sql) {
    analytics::capture_error("query_error", "query::execute");
}
```

#### Disabling Analytics

Set environment variable before running your application:

```bash
export SOCHDB_DISABLE_ANALYTICS=true
python your_app.py

# Or inline
SOCHDB_DISABLE_ANALYTICS=true ./your_binary
```

No events will be sent when disabled.

---

## 🔧 Improvements

### Documentation Updates

- Updated all version references from 0.2.9 to 0.3.1
- Updated installation guides for Python, JavaScript, Rust, Go
- Consistent versioning across all SDK documentation
- Updated benchmark metadata in README

### Build System

- **Rust:** Analytics feature enabled by default in client crates
  - `sochdb-client`, `sochdb-python`, `sochdb-grpc` include analytics
  - Added `json` feature to `ureq` for PostHog API calls
- **JavaScript:** Fixed ESM import paths for analytics module
- **Python:** Analytics as optional dependency group

---

## 🐛 Bug Fixes

- **JavaScript:** Fixed ESM import requiring `.js` extension in `database.ts`
- **Rust:** Fixed analytics payload structure for PostHog API
- **Tests:** Updated version expectations in JavaScript tests (0.3.0 → 0.3.1)

---

## 📦 Installation

### Python

```bash
pip install sochdb==0.3.1

# With analytics support
pip install sochdb[analytics]==0.3.1
```

### JavaScript / Node.js

```bash
npm install @sochdb/sochdb@0.3.1
```

### Rust

```toml
[dependencies]
sochdb-client = "0.3.1"

# Or with analytics
sochdb-core = { version = "0.3.1", features = ["analytics"] }
```

### Go

```bash
go get github.com/sochdb/sochdb-go@v0.3.1
```

---

## 🧪 Testing

All SDKs fully tested:
- ✅ Python: Analytics integration verified
- ✅ JavaScript: 74 tests passing
- ✅ Rust: 362 tests passing (1 flaky test in parallel HNSW)

Test analytics locally:
```bash
# Python
python test_error_tracking.py

# JavaScript
node test_error_tracking.js

# Rust
cd test_analytics_rust && cargo run
```

---

## 🔐 Privacy & Security

### What Data is Collected?

**When `database_opened` event fires:**
```json
{
  "event": "database_opened",
  "distinct_id": "0c628825688f52aa",
  "properties": {
    "mode": "embedded",
    "has_custom_path": true,
    "sdk": "python",
    "sdk_version": "0.3.1",
    "os": "Darwin",
    "arch": "x86_64"
  }
}
```

**When `error` event fires:**
```json
{
  "event": "error",
  "distinct_id": "0c628825688f52aa",
  "properties": {
    "error_type": "connection_error",
    "location": "database.open",
    "sdk": "python",
    "sdk_version": "0.3.1",
    "os": "Darwin",
    "arch": "x86_64"
  }
}
```

**What is NOT collected:**
- ❌ File paths
- ❌ Database names
- ❌ Query content
- ❌ Error messages
- ❌ User data
- ❌ IP addresses
- ❌ Hostnames (hashed only)

### Anonymous ID Generation

```python
# Stable hash of machine-specific info
machine_info = [hostname, os, arch, uid]
anonymous_id = sha256("|".join(machine_info))[:16]
# Example: "0c628825688f52aa"
```

Same machine always gets same ID, but ID cannot be reversed to identify the machine.

---

## 🔄 Migration Guide

### From v0.3.0 to v0.3.1

**No breaking changes.** This is a backward-compatible minor release.

**Optional:** Install analytics dependencies if you want to contribute usage data:

```bash
# Python
pip install posthog>=3.0.0

# JavaScript (already in optionalDependencies)
npm install posthog-node

# Rust (already enabled via features)
# No action needed
```

**Opt-out:** Add to your deployment configuration:

```bash
# .env file
SOCHDB_DISABLE_ANALYTICS=true

# Docker
ENV SOCHDB_DISABLE_ANALYTICS=true

# Kubernetes
env:
  - name: SOCHDB_DISABLE_ANALYTICS
    value: "true"
```

---

## 📚 Resources

- **Documentation:** https://sochdb.dev/docs/guides/analytics
- **Changelog:** [CHANGELOG.md](../CHANGELOG.md)
- **Issue Tracker:** https://github.com/sochdb/sochdb/issues
- **Privacy Policy:** All analytics code is open source in this repository

---

## 🙏 Acknowledgments

Thank you to the community for feedback on privacy-preserving telemetry design. Analytics helps us prioritize bug fixes and feature development while respecting user privacy.

---

## 📅 What's Next (v0.3.2)

- Distributed query execution
- Enhanced SQL JOIN performance
- Compression improvements for large datasets
- More granular analytics controls

---

**Full Changelog:** https://github.com/sochdb/sochdb/compare/v0.3.0...v0.3.1

---

*Released: January 4, 2026*  
*License: Apache 2.0*  
*Maintainer: @sushanthpy*
