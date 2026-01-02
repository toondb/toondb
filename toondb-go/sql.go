package toondb

import (
	"fmt"
)

// SQLEngine handles SQL query execution using the IPC client
type SQLEngine struct {
	client *IPCClient
}

// NewSQLEngine creates a new SQL engine
func NewSQLEngine(client *IPCClient) *SQLEngine {
	return &SQLEngine{client: client}
}

// Execute executes a SQL query
func (e *SQLEngine) Execute(sql string) (*SQLQueryResult, error) {
	return nil, fmt.Errorf("SQL engine implementation in progress - please use Python SDK for full SQL support in v0.2.9")
}
