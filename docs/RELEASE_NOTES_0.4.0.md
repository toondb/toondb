# SochDB v0.4.0 Release Notes

**Release Date**: January 12, 2026

## 🎉 Major Changes

### Project Rename: ToonDB → SochDB

SochDB v0.4.0 marks a significant milestone with the complete rename of the project from **ToonDB** to **SochDB**. This change affects all aspects of the project:

#### What Changed

- **Package Names**: All crates renamed from `toondb-*` to `sochdb-*`
- **Type Names**: All types updated (e.g., `ToonDB` → `SochDB`, `ToonTable` → `SochTable`, `ToonValue` → `SochValue`)
- **Module Names**: Internal modules renamed (e.g., `toon::` → `soch::`)
- **Python Package**: Renamed from `toondb` to `sochdb`
- **URLs and Domains**: Updated to `sochdb.dev` and `github.com/sochdb/sochdb`
- **Query Language**: ToonQL renamed to SochQL

#### What Stayed the Same

- **TOON Format**: The TOON serialization format name remains unchanged (similar to JSON, it's an established format name)
- **toon-format crate**: External crate dependency kept as-is
- **Core Architecture**: All the sync-first design and performance optimizations from v0.3.5 are preserved

## 📦 Migration Guide

### For Rust Users

Update your `Cargo.toml`:

```toml
# Before (v0.3.x)
[dependencies]
toondb = "0.3"
toondb-core = "0.3"

# After (v0.4.0)
[dependencies]
sochdb = "0.4"
sochdb-core = "0.4"
```

Update your imports:

```rust
// Before
use toondb::Database;
use toondb_core::toon::{ToonTable, ToonValue};

// After
use sochdb::Database;
use sochdb_core::soch::{SochTable, SochValue};
```

### For Python Users

Update your package installation:

```bash
# Before
pip install toondb

# After
pip install sochdb
```

Update your imports:

```python
# Before
import toondb
from toondb import build_index

# After
import sochdb
from sochdb import build_index
```

### For Node.js Users

Update your package.json:

```json
{
  "dependencies": {
    "sochdb": "^0.4.0"
  }
}
```

Update your requires:

```javascript
// Before
const { Database } = require('toondb');

// After
const { Database } = require('sochdb');
```

## 🔧 Technical Details

### Version Bumps

All crates updated to version 0.4.0:
- `sochdb` (formerly `toondb-client`)
- `sochdb-core` (formerly `toondb-core`)
- `sochdb-storage` (formerly `toondb-storage`)
- `sochdb-index` (formerly `toondb-index`)
- `sochdb-query` (formerly `toondb-query`)
- `sochdb-vector` (formerly `toondb-vector`)
- `sochdb-grpc` (formerly `toondb-grpc`)
- `sochdb-mcp` (formerly `toondb-mcp`)
- `sochdb-kernel` (formerly `toondb-kernel`)
- `sochdb-wasm` (formerly `toondb-wasm`)
- `sochdb-tools` (formerly `toondb-tools`)
- `sochdb-plugin-logging` (formerly `toondb-plugin-logging`)

### Build Status

- ✅ All builds passing
- ✅ 736+ tests passing across all packages
- ✅ Python bindings updated
- ✅ Documentation updated

## 📚 Documentation Updates

- README updated with rename notice
- All API documentation updated
- Migration guide added
- GitHub repository moved to `github.com/sochdb/sochdb`

## 🔗 Links

- **Repository**: https://github.com/sochdb/sochdb
- **Documentation**: https://sochdb.dev
- **Homepage**: https://sochdb.dev
- **Crates.io**: https://crates.io/crates/sochdb

## ⚠️ Breaking Changes

This is a **breaking release** due to the rename. All code using ToonDB v0.3.x will need to be updated to use SochDB v0.4.0 with the new package and type names.

## 🙏 Acknowledgments

Thank you to all contributors and users who have supported this project. The rename to SochDB represents our continued commitment to building the best AI-native database for LLM applications.
