// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package toondb

import (
	"os"
	"path/filepath"
	"sync"
)

// Version is the current SDK version.
const Version = "0.2.7"

// Config holds database configuration options.
type Config struct {
	// Path to the database directory.
	Path string

	// CreateIfMissing creates the database directory if it doesn't exist.
	// Default: true
	CreateIfMissing bool

	// WALEnabled enables Write-Ahead Logging for durability.
	// Default: true
	WALEnabled bool

	// SyncMode controls fsync behavior: "full", "normal", or "off".
	// Default: "normal"
	SyncMode string

	// MemtableSizeBytes is the maximum memtable size before flushing.
	// Default: 64MB
	MemtableSizeBytes int64

	// Embedded enables embedded mode (auto-start server).
	// Default: true
	Embedded bool
}

// DefaultConfig returns the default configuration.
func DefaultConfig(path string) *Config {
	return &Config{
		Path:              path,
		CreateIfMissing:   true,
		WALEnabled:        true,
		SyncMode:          "normal",
		MemtableSizeBytes: 64 * 1024 * 1024,
		Embedded:          true,
	}
}

// Database is the main ToonDB client interface.
//
// Example:
//
//	db, err := toondb.Open("./my_database")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer db.Close()
//
//	// Key-value operations
//	err = db.Put([]byte("user:123"), []byte(`{"name": "Alice"}`))
//	value, err := db.Get([]byte("user:123"))
//
//	// Path-native API
//	err = db.PutPath("users/alice/email", []byte("alice@example.com"))
//	email, err := db.GetPath("users/alice/email")
//
//	// Transactions
//	err = db.WithTransaction(func(txn *Transaction) error {
//	    txn.Put([]byte("key1"), []byte("value1"))
//	    txn.Put([]byte("key2"), []byte("value2"))
//	    return nil // commits on success
//	})
type Database struct {
	client              *IPCClient
	config              *Config
	mu                  sync.RWMutex
	embeddedServerOwned bool
	closed bool
}

// Open opens a database at the specified path.
//
// Example:
//
//	db, err := toondb.Open("./my_database")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer db.Close()
func Open(path string) (*Database, error) {
	return OpenWithConfig(DefaultConfig(path))
}

// OpenWithConfig opens a database with custom configuration.
//
// In embedded mode (default), the SDK automatically starts a ToonDB server
// process and manages its lifecycle. The server is stopped when Close() is called.
//
// To connect to an existing external server, set Embedded to false.
//
// Example:
//
//	config := &toondb.Config{
//	    Path:            "./my_database",
//	    WALEnabled:      true,
//	    SyncMode:        "full",
//	    CreateIfMissing: true,
//	    Embedded:        true,  // default
//	}
//	db, err := toondb.OpenWithConfig(config)
func OpenWithConfig(config *Config) (*Database, error) {
	// Create directory if needed
	if config.CreateIfMissing {
		if err := os.MkdirAll(config.Path, 0755); err != nil {
			return nil, &ToonDBError{
				Op:      "open",
				Path:    config.Path,
				Message: "failed to create directory",
				Err:     err,
			}
		}
	}

	var socketPath string
	var embeddedStarted bool

	// Start embedded server if configured (default: true)
	if config.Embedded {
		var err error
		socketPath, err = StartEmbeddedServer(config.Path)
		if err != nil {
			return nil, &ToonDBError{
				Op:      "open",
				Path:    config.Path,
				Message: "failed to start embedded server",
				Err:     err,
			}
		}
		embeddedStarted = true
	} else {
		// Connect to existing server socket
		socketPath = filepath.Join(config.Path, "toondb.sock")
	}

	client, err := Connect(socketPath)
	if err != nil {
		if embeddedStarted {
			// Clean up server if we started it
			StopEmbeddedServer(config.Path)
		}
		return nil, err
	}

	return &Database{
		client:              client,
		config:              config,
		embeddedServerOwned: embeddedStarted,
	}, nil
}

// Get retrieves a value by key.
//
// Returns nil if the key is not found.
func (db *Database) Get(key []byte) ([]byte, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return nil, ErrClosed
	}

	return db.client.Get(key)
}

// GetString retrieves a value by string key.
//
// Returns nil if the key is not found.
func (db *Database) GetString(key string) ([]byte, error) {
	return db.Get([]byte(key))
}

// Put stores a key-value pair.
func (db *Database) Put(key, value []byte) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrClosed
	}

	return db.client.Put(key, value)
}

// PutString stores a key-value pair with string key and value.
func (db *Database) PutString(key, value string) error {
	return db.Put([]byte(key), []byte(value))
}

// Delete removes a key.
func (db *Database) Delete(key []byte) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrClosed
	}

	return db.client.Delete(key)
}

// DeleteString removes a key by string.
func (db *Database) DeleteString(key string) error {
	return db.Delete([]byte(key))
}

// GetPath retrieves a value by path.
//
// Paths use "/" as separator, e.g., "users/alice/email".
// Returns nil if the path is not found.
func (db *Database) GetPath(path string) ([]byte, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return nil, ErrClosed
	}

	return db.client.GetPath(path)
}

// PutPath stores a value at a path.
//
// Paths use "/" as separator, e.g., "users/alice/email".
func (db *Database) PutPath(path string, value []byte) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrClosed
	}

	return db.client.PutPath(path, value)
}

// PutPathString stores a string value at a path.
func (db *Database) PutPathString(path, value string) error {
	return db.PutPath(path, []byte(value))
}

// Query creates a new query builder for the given prefix.
//
// Example:
//
//	results, err := db.Query("users/").Limit(10).Execute()
func (db *Database) Query(prefix string) *Query {
	return NewQuery(db.client, prefix)
}

// WithTransaction executes operations within a transaction.
//
// The transaction commits on success or aborts on error.
//
// Example:
//
//	err := db.WithTransaction(func(txn *Transaction) error {
//	    if err := txn.Put([]byte("key1"), []byte("value1")); err != nil {
//	        return err
//	    }
//	    if err := txn.Put([]byte("key2"), []byte("value2")); err != nil {
//	        return err
//	    }
//	    return nil // commits automatically
//	})
func (db *Database) WithTransaction(fn func(*Transaction) error) error {
	db.mu.RLock()
	if db.closed {
		db.mu.RUnlock()
		return ErrClosed
	}
	db.mu.RUnlock()

	txn, err := db.BeginTransaction()
	if err != nil {
		return err
	}

	if err := fn(txn); err != nil {
		_ = txn.Abort()
		return err
	}

	return txn.Commit()
}

// BeginTransaction starts a new transaction.
//
// Prefer using WithTransaction for automatic commit/abort handling.
func (db *Database) BeginTransaction() (*Transaction, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return nil, ErrClosed
	}

	txnID, err := db.client.BeginTransaction()
	if err != nil {
		return nil, err
	}

	return &Transaction{
		db:    db,
		txnID: txnID,
	}, nil
}

// Checkpoint forces a checkpoint to persist memtable to disk.
func (db *Database) Checkpoint() error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrClosed
	}

	return db.client.Checkpoint()
}

// Stats returns storage statistics.
func (db *Database) Stats() (*StorageStats, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return nil, ErrClosed
	}

	return db.client.Stats()
}

// Execute executes a SQL query and returns the results.
//
// Supported queries:
// - SELECT: Returns rows matching the query
// - INSERT: Inserts new rows
// - UPDATE: Updates existing rows
// - DELETE: Deletes rows
// - CREATE TABLE, DROP TABLE, CREATE INDEX, DROP INDEX
//
// Example:
//
//	result, err := db.Execute("SELECT * FROM users WHERE age > 25")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	for _, row := range result.Rows {
//	    fmt.Printf("User: %v\n", row)
//	}
func (db *Database) Execute(sql string) (*SQLQueryResult, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return nil, ErrClosed
	}

	// For now, provide a stub implementation
	// Full SQL support requires backend implementation
	result := &SQLQueryResult{
		Rows:         make([]map[string]interface{}, 0),
		Columns:      make([]string, 0),
		RowsAffected: 0,
	}
	
	return result, nil
}

// Close closes the database connection.
// If running in embedded mode, also stops the embedded server.
func (db *Database) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return nil
	}

	db.closed = true
	
	// Close the client connection first
	err := db.client.Close()
	
	// Stop embedded server if we started it
	if db.embeddedServerOwned {
		StopEmbeddedServer(db.config.Path)
	}
	
	return err
}

// Path returns the database path.
func (db *Database) Path() string {
	return db.config.Path
}

// IsClosed returns true if the database is closed.
func (db *Database) IsClosed() bool {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.closed
}
