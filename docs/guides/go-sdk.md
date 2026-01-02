# Go SDK Guide

> **🔧 Skill Level:** Beginner  
> **⏱️ Time Required:** 20 minutes  
> **📦 Requirements:** Go 1.21+

Complete guide to ToonDB's Go SDK with all features, patterns, and best practices.

---

## 📦 Installation

```bash
go get github.com/toondb/toondb/toondb-go@v0.2.9
```

**What's New in 0.2.7:**
- ✅ Embedded server mode with automatic lifecycle management
- ✅ Zero external setup required
- ✅ Platform-specific binary discovery

**What's New in 0.2.6:**
- ✅ Fixed wire protocol compatibility (Little Endian)
- ✅ Added `Scan()` method for prefix-based iteration
- ✅ Fixed `GetPath` and `PutPath` encoding
- ✅ Improved error handling
- ✅ Zero CGO dependencies

---

## CLI Tools (v0.2.9+)

ToonDB includes Go-native wrappers for installation via `go install`:

1. **`toondb-server`**: IPC server management.
   ```bash
   go install github.com/toondb/toondb/toondb-go/cmd/toondb-server@latest
   ```

2. **`toondb-bulk`**: Bulk operations tool.
   ```bash
   go install github.com/toondb/toondb/toondb-go/cmd/toondb-bulk@latest
   ```

3. **`toondb-grpc-server`**: gRPC vector server.
   ```bash
   go install github.com/toondb/toondb/toondb-go/cmd/toondb-grpc-server@latest
   ```

> **Deep Dive:** See [Server Reference](/servers/IPC_SERVER.md) for full usage.

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    toondb "github.com/toondb/toondb/toondb-go"
)

func main() {
    // Open database (creates if doesn't exist)
    db, err := toondb.Open("./my_database")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Put and Get
    err = db.Put([]byte("user:123"), []byte(`{"name":"Alice","age":30}`))
    if err != nil {
        log.Fatal(err)
    }

    value, err := db.Get([]byte("user:123"))
    if err != nil {
        log.Fatal(err)
    }
    
    if value != nil {
        fmt.Println(string(value))
        // Output: {"name":"Alice","age":30}
    }
}
```

**Output:**
```
{"name":"Alice","age":30}
```

---

## Core Operations

### Put Operation

Store key-value pairs:

```go
// Binary key-value
err := db.Put([]byte("key"), []byte("value"))

// String helper
err = db.PutString("greeting", "Hello World")

// JSON data
import "encoding/json"

user := map[string]interface{}{
    "name": "Alice",
    "email": "alice@example.com",
    "age": 30,
}
data, _ := json.Marshal(user)
err = db.Put([]byte("users/alice"), data)
```

### Get Operation

Retrieve values:

```go
// Get with error handling
value, err := db.Get([]byte("key"))
if err != nil {
    log.Fatal(err)
}

if value == nil {
    fmt.Println("Key not found")
} else {
    fmt.Println(string(value))
}

// String helper
str, err := db.GetString("greeting")
if str != nil {
    fmt.Println(string(str))
}

// Parse JSON
var user map[string]interface{}
if value != nil {
    json.Unmarshal(value, &user)
    fmt.Printf("Name: %s\n", user["name"])
}
```

### Delete Operation

Remove keys:

```go
err := db.Delete([]byte("key"))
if err != nil {
    log.Fatal(err)
}

// String helper
err = db.DeleteString("greeting")
```

---

## Path API

ToonDB treats hierarchical paths as first-class citizens:

```go
// Store hierarchical data
db.PutPath("users/alice/email", []byte("alice@example.com"))
db.PutPath("users/alice/age", []byte("30"))
db.PutPath("users/alice/settings/theme", []byte("dark"))
db.PutPath("users/bob/email", []byte("bob@example.com"))

// Retrieve by path
email, err := db.GetPath("users/alice/email")
if email != nil {
    fmt.Printf("Alice's email: %s\n", email)
}

// Delete by path
err = db.DeletePath("users/alice/settings/theme")
```

**Output:**
```
Alice's email: alice@example.com
```

**Path Format:**
- Paths are separated by `/`
- Automatically encoded to binary keys
- O(|path|) lookup performance

---

## Prefix Scanning

⭐ **New in 0.2.6** - The most efficient way to iterate keys:

### Basic Scan

```go
// Insert data
db.Put([]byte("users/1"), []byte(`{"name":"Alice"}`))
db.Put([]byte("users/2"), []byte(`{"name":"Bob"}`))
db.Put([]byte("orders/1"), []byte(`{"total":100}`))

// Scan all keys with "users/" prefix
results, err := db.Scan("users/")
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Found %d users:\n", len(results))
for _, kv := range results {
    fmt.Printf("  %s: %s\n", kv.Key, kv.Value)
}
```

**Output:**
```
Found 2 users:
  users/1: {"name":"Alice"}
  users/2: {"name":"Bob"}
```

### Multi-Tenant Data Isolation

```go
// Insert tenant-specific data
db.Put([]byte("tenants/acme/users/1"), []byte(`{"name":"Alice"}`))
db.Put([]byte("tenants/acme/users/2"), []byte(`{"name":"Bob"}`))
db.Put([]byte("tenants/acme/orders/1"), []byte(`{"total":100}`))
db.Put([]byte("tenants/globex/users/1"), []byte(`{"name":"Charlie"}`))

// Scan only ACME Corp data (tenant isolation)
acmeData, _ := db.Scan("tenants/acme/")
fmt.Printf("ACME Corp: %d items\n", len(acmeData))

// Scan only Globex Corp data
globexData, _ := db.Scan("tenants/globex/")
fmt.Printf("Globex Corp: %d items\n", len(globexData))
```

**Output:**
```
ACME Corp: 3 items
Globex Corp: 1 items
```

**Why use Scan():**
- **Fast**: O(|prefix|) performance
- **Isolated**: Perfect for multi-tenant apps
- **Efficient**: No deserialization overhead
- **Binary-safe**: Works with any key format

---

## Transactions

### Automatic Transactions

```go
// Context manager pattern
err := db.WithTransaction(func(txn *toondb.Transaction) error {
    // All operations are atomic
    txn.Put([]byte("account:1:balance"), []byte("1000"))
    txn.Put([]byte("account:2:balance"), []byte("500"))
    
    // Automatic commit on return nil
    // Automatic abort on return error
    return nil
})

if err != nil {
    log.Printf("Transaction failed: %v", err)
}
```

**Output:**
```
✅ Transaction committed
```

### Manual Transaction Control

```go
txn, err := db.BeginTransaction()
if err != nil {
    log.Fatal(err)
}
defer txn.Abort() // Safety net

// Perform operations
txn.Put([]byte("key1"), []byte("value1"))
txn.Put([]byte("key2"), []byte("value2"))

// Read within transaction
value, _ := txn.Get([]byte("key1"))
fmt.Println(string(value))

// Commit if successful
err = txn.Commit()
if err != nil {
    log.Printf("Commit failed: %v", err)
}
```

### Transaction Isolation

```go
// Snapshot isolation example
txn1, _ := db.BeginTransaction()
txn2, _ := db.BeginTransaction()

// txn1 writes
txn1.Put([]byte("counter"), []byte("100"))

// txn2 reads old value (isolation)
val, _ := txn2.Get([]byte("counter"))
fmt.Println(string(val)) // Original value

txn1.Commit()

// Now txn2 sees updated value after commit
val2, _ := db.Get([]byte("counter"))
fmt.Println(string(val2)) // "100"
```

---

## Query Builder

Returns results in **TOON format** (token-optimized for LLMs):

### Basic Query

```go
// Insert structured data
db.Put([]byte("products/laptop"), []byte(`{"name":"Laptop","price":999,"stock":5}`))
db.Put([]byte("products/mouse"), []byte(`{"name":"Mouse","price":25,"stock":20}`))
db.Put([]byte("products/keyboard"), []byte(`{"name":"Keyboard","price":75,"stock":10}`))

// Query with column selection
results, err := db.Query("products/").
    Select("name", "price").
    Limit(10).
    Execute()

if err != nil {
    log.Fatal(err)
}

for _, kv := range results {
    fmt.Printf("%s: %s\n", kv.Key, kv.Value)
}
```

**Output (TOON Format):**
```
products/laptop: result[1]{name,price}:Laptop,999
products/mouse: result[1]{name,price}:Mouse,25
products/keyboard: result[1]{name,price}:Keyboard,75
```

### Query Methods

```go
// Get first result
first, err := db.Query("products/").First()
if first != nil {
    fmt.Printf("First: %s\n", first.Value)
}

// Count results
count, err := db.Query("products/").Count()
fmt.Printf("Total products: %d\n", count)

// Check existence
exists, err := db.Query("products/").Exists()
fmt.Printf("Products exist: %v\n", exists)

// With filters
results, err := db.Query("products/").
    Select("name", "price", "stock").
    Limit(5).
    Execute()
```

### SQL-Like Operations

**✅ SQL Support Now Available (v0.2.9+):**

```go
// Create table
_, err := db.Execute(`
    CREATE TABLE users (
        id INT PRIMARY KEY,
        name TEXT,
        email TEXT,
        age INT
    )
`)

// Insert data
_, err = db.Execute("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com', 30)")

// Query data
result, err := db.Execute("SELECT * FROM users WHERE age > 25")
for _, row := range result.Rows {
    fmt.Printf("User: %v\n", row)
}
```

**Or use the Query Builder for JSON documents:**

```go
// INSERT-like: Store structured data
type Product struct {
    ID    int     `json:"id"`
    Name  string  `json:"name"`
    Price float64 `json:"price"`
}

product := Product{ID: 1, Name: "Laptop", Price: 999.99}
data, _ := json.Marshal(product)
db.Put([]byte("products/001"), data)

// SELECT-like: Query with column selection
results, _ := db.Query("products/").
    Select("name", "price").  // SELECT name, price
    Limit(10).                 // LIMIT 10
    Execute()

for _, kv := range results {
    fmt.Printf("%s: %s\n", kv.Key, kv.Value)
}
```

---

## Vector Search

### HNSW Index

```go
import "github.com/toondb/toondb/toondb-index"

// Create HNSW index
config := &toondb.VectorIndexConfig{
    Dimension:      384,
    Metric:         toondb.Cosine,
    M:              16,
    EfConstruction: 100,
}

index := toondb.NewVectorIndex("./my_index", config)

// Build from embeddings
embeddings := [][]float32{
    {0.1, 0.2, 0.3, /* ... 384 dimensions */},
    {0.4, 0.5, 0.6, /* ... 384 dimensions */},
}
labels := []string{"doc1", "doc2"}

err := index.BulkBuild(embeddings, labels)
if err != nil {
    log.Fatal(err)
}

// Search
query := []float32{0.15, 0.25, 0.35, /* ... */}
results, err := index.Query(query, 10, 50) // k=10, ef_search=50

for i, r := range results {
    fmt.Printf("%d. %s (distance: %.4f)\n", i+1, r.Label, r.Distance)
}
```

**Output:**
```
1. doc1 (distance: 0.0234)
2. doc2 (distance: 0.1567)
```

### Distance Metrics

```go
// Cosine similarity
config.Metric = toondb.Cosine

// Euclidean distance
config.Metric = toondb.Euclidean

// Dot product
config.Metric = toondb.DotProduct
```

---

## Error Handling

### Standard Error Checking

```go
import "errors"

value, err := db.Get([]byte("key"))
if err != nil {
    // Check for specific errors
    if errors.Is(err, toondb.ErrClosed) {
        log.Println("Database is closed")
    } else if errors.Is(err, toondb.ErrConnectionFailed) {
        log.Println("Cannot connect to server")
    } else {
        log.Printf("Unknown error: %v", err)
    }
    return err
}

// Check for missing key (not an error)
if value == nil {
    log.Println("Key not found")
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `ErrClosed` | Database/connection closed |
| `ErrConnectionFailed` | Cannot connect to server |
| `ErrTimeout` | Operation timed out |
| `ErrInvalidKey` | Key format invalid |
| `ErrTransactionConflict` | Transaction conflict |

### Panic Recovery

```go
func safeOperation(db *toondb.Database) (err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()
    
    // Your operations here
    db.Put([]byte("key"), []byte("value"))
    return nil
}
```

---

## Best Practices

### 1. Always Close Resources

```go
db, err := toondb.Open("./my_database")
if err != nil {
    log.Fatal(err)
}
defer db.Close() // Always defer close
```

### 2. Use Transactions for Atomic Operations

```go
// ❌ Bad: Not atomic
db.Put([]byte("balance:1"), []byte("900"))
db.Put([]byte("balance:2"), []byte("1100"))

// ✅ Good: Atomic
db.WithTransaction(func(txn *toondb.Transaction) error {
    txn.Put([]byte("balance:1"), []byte("900"))
    txn.Put([]byte("balance:2"), []byte("1100"))
    return nil
})
```

### 3. Check Nil for Missing Keys

```go
value, err := db.Get([]byte("key"))
if err != nil {
    return err
}

if value == nil {
    // Key doesn't exist (not an error!)
    return nil
}

// Process value
fmt.Println(string(value))
```

### 4. Use Scan() for Prefix Iteration

```go
// ❌ Bad: Query returns TOON format
results, _ := db.Query("users/").Execute()

// ✅ Good: Scan returns raw binary
results, _ := db.Scan("users/")
```

### 5. Prefix Keys for Multi-Tenancy

```go
// Store tenant-specific data
tenantID := "acme"
userKey := fmt.Sprintf("tenants/%s/users/%s", tenantID, userID)
db.Put([]byte(userKey), userData)

// Scan tenant data only
prefix := fmt.Sprintf("tenants/%s/", tenantID)
data, _ := db.Scan(prefix)
```

### 6. Handle Errors Properly

```go
// ❌ Bad: Ignoring errors
value, _ := db.Get([]byte("key"))

// ✅ Good: Proper error handling
value, err := db.Get([]byte("key"))
if err != nil {
    log.Printf("Get failed: %v", err)
    return err
}
```

---

## Complete Examples

### Example 1: User Management System

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    toondb "github.com/toondb/toondb/toondb-go"
)

type User struct {
    ID    string `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

func main() {
    db, _ := toondb.Open("./users_db")
    defer db.Close()

    // Create user
    user := User{
        ID:    "alice",
        Name:  "Alice Smith",
        Email: "alice@example.com",
    }
    
    data, _ := json.Marshal(user)
    db.Put([]byte("users/"+user.ID), data)

    // Get user
    value, _ := db.Get([]byte("users/alice"))
    if value != nil {
        var u User
        json.Unmarshal(value, &u)
        fmt.Printf("User: %s (%s)\n", u.Name, u.Email)
    }

    // List all users
    results, _ := db.Scan("users/")
    fmt.Printf("Total users: %d\n", len(results))
}
```

### Example 2: Multi-Tenant SaaS Application

```go
package main

import (
    "encoding/json"
    "fmt"
    toondb "github.com/toondb/toondb/toondb-go"
)

type TenantData struct {
    Role  string `json:"role"`
    Email string `json:"email"`
}

func main() {
    db, _ := toondb.Open("./saas_db")
    defer db.Close()

    // Store tenant data
    tenants := map[string][]struct {
        UserID string
        Data   TenantData
    }{
        "acme": {
            {"alice", TenantData{"admin", "alice@acme.com"}},
            {"bob", TenantData{"user", "bob@acme.com"}},
        },
        "globex": {
            {"charlie", TenantData{"admin", "charlie@globex.com"}},
        },
    }

    for tenantID, users := range tenants {
        for _, u := range users {
            key := fmt.Sprintf("tenants/%s/users/%s", tenantID, u.UserID)
            data, _ := json.Marshal(u.Data)
            db.Put([]byte(key), data)
        }
    }

    // Query ACME Corp data only (isolation)
    acmeData, _ := db.Scan("tenants/acme/")
    fmt.Printf("ACME Corp: %d users\n", len(acmeData))
    
    for _, kv := range acmeData {
        var td TenantData
        json.Unmarshal(kv.Value, &td)
        fmt.Printf("  %s: %s (%s)\n", kv.Key, td.Email, td.Role)
    }

    // Query Globex Corp data
    globexData, _ := db.Scan("tenants/globex/")
    fmt.Printf("Globex Corp: %d users\n", len(globexData))
}
```

**Output:**
```
ACME Corp: 2 users
  tenants/acme/users/alice: alice@acme.com (admin)
  tenants/acme/users/bob: bob@acme.com (user)
Globex Corp: 1 users
```

### Example 3: Session Cache

```go
package main

import (
    "encoding/json"
    "fmt"
    "time"
    toondb "github.com/toondb/toondb/toondb-go"
)

type Session struct {
    UserID    string    `json:"user_id"`
    Token     string    `json:"token"`
    ExpiresAt time.Time `json:"expires_at"`
}

func main() {
    db, _ := toondb.Open("./sessions")
    defer db.Close()

    // Create session
    session := Session{
        UserID:    "alice",
        Token:     "abc123",
        ExpiresAt: time.Now().Add(24 * time.Hour),
    }
    
    data, _ := json.Marshal(session)
    db.Put([]byte("session:"+session.Token), data)

    // Get session
    value, _ := db.Get([]byte("session:abc123"))
    if value != nil {
        var s Session
        json.Unmarshal(value, &s)
        
        if time.Now().Before(s.ExpiresAt) {
            fmt.Printf("Valid session for user: %s\n", s.UserID)
        } else {
            fmt.Println("Session expired")
            db.Delete([]byte("session:" + s.Token))
        }
    }
}
```

---

## API Reference

### Database

| Method | Signature | Description |
|--------|-----------|-------------|
| `Open(path)` | `func Open(path string) (*Database, error)` | Open/create database |
| `Close()` | `func (db *Database) Close() error` | Close database |
| `Put(key, value)` | `func (db *Database) Put(key, value []byte) error` | Store key-value |
| `Get(key)` | `func (db *Database) Get(key []byte) ([]byte, error)` | Retrieve value |
| `Delete(key)` | `func (db *Database) Delete(key []byte) error` | Delete key |
| `PutPath(path, value)` | `func (db *Database) PutPath(path string, value []byte) error` | Store by path |
| `GetPath(path)` | `func (db *Database) GetPath(path string) ([]byte, error)` | Get by path |
| `Scan(prefix)` | `func (db *Database) Scan(prefix string) ([]KeyValue, error)` | Scan prefix ⭐ New |
| `BeginTransaction()` | `func (db *Database) BeginTransaction() (*Transaction, error)` | Start transaction |
| `Query(prefix)` | `func (db *Database) Query(prefix string) *QueryBuilder` | Create query |

### Transaction

| Method | Signature | Description |
|--------|-----------|-------------|
| `Put(key, value)` | `func (txn *Transaction) Put(key, value []byte) error` | Store in transaction |
| `Get(key)` | `func (txn *Transaction) Get(key []byte) ([]byte, error)` | Get in transaction |
| `Delete(key)` | `func (txn *Transaction) Delete(key []byte) error` | Delete in transaction |
| `Commit()` | `func (txn *Transaction) Commit() error` | Commit transaction |
| `Abort()` | `func (txn *Transaction) Abort() error` | Abort transaction |

### QueryBuilder

| Method | Signature | Description |
|--------|-----------|-------------|
| `Select(cols...)` | `func (q *QueryBuilder) Select(cols ...string) *QueryBuilder` | Select columns |
| `Limit(n)` | `func (q *QueryBuilder) Limit(n int) *QueryBuilder` | Limit results |
| `Execute()` | `func (q *QueryBuilder) Execute() ([]KeyValue, error)` | Execute query |
| `First()` | `func (q *QueryBuilder) First() (*KeyValue, error)` | Get first result |
| `Count()` | `func (q *QueryBuilder) Count() (int, error)` | Count results |
| `Exists()` | `func (q *QueryBuilder) Exists() (bool, error)` | Check existence |

---

## Configuration

```go
config := &toondb.Config{
    Path:              "./my_database",
    CreateIfMissing:   true,
    WALEnabled:        true,
    SyncMode:          "normal", // "full", "normal", "off"
    MemtableSizeBytes: 64 * 1024 * 1024,
}

db, err := toondb.OpenWithConfig(config)
```

---

## Testing

```bash
# Run tests
go test -v ./...

# Run benchmarks
go test -bench=. -benchmem ./...

# Run specific test
go test -run TestScan -v
```

---

## Performance Tips

1. **Batch writes in transactions**: Up to 10× faster
2. **Use Scan() for iteration**: Faster than Query()
3. **Reuse Database connections**: Avoid repeated Open/Close
4. **Use binary keys when possible**: Faster than string conversion
5. **Profile with pprof**: `import _ "net/http/pprof"`

---

## Resources

- [Go SDK GitHub](https://github.com/toondb/toondb/tree/main/toondb-go)
- [API Reference](../api-reference/go-api.md)
- [Python SDK](./python-sdk.md)
- [JavaScript SDK](./nodejs-sdk.md)
- [Rust SDK](./rust-sdk.md)

---

*Last updated: January 2026 (v0.2.9)*
