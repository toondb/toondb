// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package toondb

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"path/filepath"
	"sync"
)

// OpCode represents the wire protocol operation codes.
// Must match toondb-storage/src/ipc_server.rs opcodes exactly.
type OpCode uint8

// Client → Server opcodes
const (
	OpPut        OpCode = 0x01
	OpGet        OpCode = 0x02
	OpDelete     OpCode = 0x03
	OpBeginTxn   OpCode = 0x04
	OpCommitTxn  OpCode = 0x05
	OpAbortTxn   OpCode = 0x06
	OpQuery      OpCode = 0x07
	OpPutPath    OpCode = 0x09
	OpGetPath    OpCode = 0x0A
	OpScan       OpCode = 0x0B
	OpCheckpoint OpCode = 0x0C
	OpStats      OpCode = 0x0D
	OpPing       OpCode = 0x0E
	OpExecuteSQL OpCode = 0x0F
)

// Server → Client response opcodes
const (
	OpOK        OpCode = 0x80
	OpError     OpCode = 0x81
	OpValue     OpCode = 0x82
	OpTxnID     OpCode = 0x83
	OpRow       OpCode = 0x84
	OpEndStream OpCode = 0x85
	OpStatsResp OpCode = 0x86
	OpPong      OpCode = 0x87
)

// IPCClient handles low-level IPC communication with the ToonDB server.
type IPCClient struct {
	conn   net.Conn
	mu     sync.Mutex
	closed bool
}

// Connect establishes a connection to the ToonDB server.
func Connect(socketPath string) (*IPCClient, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, &ConnectionError{Address: socketPath, Err: err}
	}
	return &IPCClient{conn: conn}, nil
}

// ConnectToDatabase connects to a database at the given path.
func ConnectToDatabase(dbPath string) (*IPCClient, error) {
	socketPath := filepath.Join(dbPath, "toondb.sock")
	return Connect(socketPath)
}

// Close closes the connection.
func (c *IPCClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}
	c.closed = true
	return c.conn.Close()
}

// Get retrieves a value by key.
func (c *IPCClient) Get(key []byte) ([]byte, error) {
	return c.sendKeyOp(OpGet, key)
}

// Put stores a key-value pair.
func (c *IPCClient) Put(key, value []byte) error {
	_, err := c.sendKeyValueOp(OpPut, key, value)
	return err
}

// Delete removes a key.
func (c *IPCClient) Delete(key []byte) error {
	_, err := c.sendKeyOp(OpDelete, key)
	return err
}

// GetPath retrieves a value by path.
func (c *IPCClient) GetPath(path string) ([]byte, error) {
	return c.sendKeyOp(OpGetPath, []byte(path))
}

// PutPath stores a value at a path.
func (c *IPCClient) PutPath(path string, value []byte) error {
	_, err := c.sendKeyValueOp(OpPutPath, []byte(path), value)
	return err
}

// Scan scans keys with a prefix, returning key-value pairs.
// This is the preferred method for prefix-based iteration.
func (c *IPCClient) Scan(prefix string) ([]KeyValue, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, ErrClosed
	}

	// SCAN payload is just the prefix string
	prefixBytes := []byte(prefix)
	if err := c.sendMessage(OpScan, prefixBytes); err != nil {
		return nil, err
	}

	return c.readScanResponse()
}

// Query executes a prefix query.
// Wire format: path_len(2 LE) + path + limit(4 LE) + offset(4 LE) + cols_count(2 LE)
func (c *IPCClient) Query(prefix string, limit, offset int) ([]KeyValue, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, ErrClosed
	}

	// Build query payload
	prefixBytes := []byte(prefix)
	// Format: path_len(2) + path + limit(4) + offset(4) + cols_count(2)
	payloadLen := 2 + len(prefixBytes) + 4 + 4 + 2
	payload := make([]byte, payloadLen)
	binary.LittleEndian.PutUint16(payload[0:2], uint16(len(prefixBytes)))
	copy(payload[2:2+len(prefixBytes)], prefixBytes)
	binary.LittleEndian.PutUint32(payload[2+len(prefixBytes):6+len(prefixBytes)], uint32(limit))
	binary.LittleEndian.PutUint32(payload[6+len(prefixBytes):10+len(prefixBytes)], uint32(offset))
	binary.LittleEndian.PutUint16(payload[10+len(prefixBytes):12+len(prefixBytes)], 0) // no columns

	// Send message: opcode(1) + length(4 LE) + payload
	if err := c.sendMessage(OpQuery, payload); err != nil {
		return nil, err
	}

	// Read response
	return c.readQueryResponse()
}

// BeginTransaction starts a new transaction.
func (c *IPCClient) BeginTransaction() (uint64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return 0, ErrClosed
	}

	if err := c.sendMessage(OpBeginTxn, nil); err != nil {
		return 0, err
	}

	// Read response
	opcode, payload, err := c.readMessage()
	if err != nil {
		return 0, err
	}

	if opcode != OpTxnID {
		if opcode == OpError {
			return 0, c.parseErrorPayload(payload)
		}
		return 0, &ProtocolError{Message: fmt.Sprintf("expected TXN_ID, got opcode %#x", opcode)}
	}

	if len(payload) < 8 {
		return 0, &ProtocolError{Message: "invalid transaction response"}
	}

	txnID := binary.LittleEndian.Uint64(payload[0:8])
	return txnID, nil
}

// CommitTransaction commits a transaction.
func (c *IPCClient) CommitTransaction(txnID uint64) error {
	return c.sendTxnOp(OpCommitTxn, txnID)
}

// AbortTransaction aborts a transaction.
func (c *IPCClient) AbortTransaction(txnID uint64) error {
	return c.sendTxnOp(OpAbortTxn, txnID)
}

// Checkpoint forces a checkpoint.
func (c *IPCClient) Checkpoint() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrClosed
	}

	if err := c.sendMessage(OpCheckpoint, nil); err != nil {
		return err
	}

	return c.readSimpleResponse()
}

// Stats retrieves storage statistics.
func (c *IPCClient) Stats() (*StorageStats, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, ErrClosed
	}

	if err := c.sendMessage(OpStats, nil); err != nil {
		return nil, err
	}

	// Read response
	opcode, payload, err := c.readMessage()
	if err != nil {
		return nil, err
	}

	if opcode == OpError {
		return nil, c.parseErrorPayload(payload)
	}

	if opcode != OpStatsResp && opcode != OpOK {
		return nil, &ProtocolError{Message: fmt.Sprintf("unexpected stats response opcode: %#x", opcode)}
	}

	// Stats response format is JSON
	if len(payload) == 0 {
		// Empty response
		return &StorageStats{}, nil
	}

	// Parse JSON response
	var stats struct {
		MemtableSizeBytes  uint64 `json:"memtable_size_bytes"`
		WALSizeBytes       uint64 `json:"wal_size_bytes"`
		ActiveTransactions int    `json:"active_transactions"`
	}

	if err := json.Unmarshal(payload, &stats); err != nil {
		return nil, &ProtocolError{Message: fmt.Sprintf("failed to parse stats JSON: %v", err)}
	}

	return &StorageStats{
		MemtableSizeBytes:  stats.MemtableSizeBytes,
		WALSizeBytes:       stats.WALSizeBytes,
		ActiveTransactions: stats.ActiveTransactions,
	}, nil
}

// ============================================================================
// Low-level Wire Protocol Helpers
// ============================================================================

// sendMessage sends a message using the wire protocol format:
// opcode(1) + length(4 LE) + payload
func (c *IPCClient) sendMessage(op OpCode, payload []byte) error {
	// Build message: [opcode:1][length:4 LE][payload:N]
	msg := make([]byte, 5+len(payload))
	msg[0] = byte(op)
	binary.LittleEndian.PutUint32(msg[1:5], uint32(len(payload)))
	if len(payload) > 0 {
		copy(msg[5:], payload)
	}

	_, err := c.conn.Write(msg)
	return err
}

// readMessage reads a message using the wire protocol format:
// opcode(1) + length(4 LE) + payload
func (c *IPCClient) readMessage() (OpCode, []byte, error) {
	// Read opcode (1 byte)
	opcodeBuf := make([]byte, 1)
	if _, err := io.ReadFull(c.conn, opcodeBuf); err != nil {
		return 0, nil, err
	}

	// Read length (4 bytes LE)
	lenBuf := make([]byte, 4)
	if _, err := io.ReadFull(c.conn, lenBuf); err != nil {
		return 0, nil, err
	}
	payloadLen := binary.LittleEndian.Uint32(lenBuf)

	// Read payload
	payload := make([]byte, payloadLen)
	if payloadLen > 0 {
		if _, err := io.ReadFull(c.conn, payload); err != nil {
			return 0, nil, err
		}
	}

	return OpCode(opcodeBuf[0]), payload, nil
}

// parseErrorPayload parses an error payload (the payload part of an ERROR response)
func (c *IPCClient) parseErrorPayload(payload []byte) error {
	msg := string(payload)
	return &ToonDBError{Op: "remote", Message: msg}
}

// Helper methods

func (c *IPCClient) sendKeyOp(op OpCode, key []byte) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, ErrClosed
	}

	// For GET/DELETE: payload is just the raw key
	// For GET_PATH: payload is path_count(2) + [seg_len(2) + seg]...
	if op == OpGetPath {
		// Path format: count(2) + [len(2) + segment]...
		path := string(key)
		payload := make([]byte, 2+2+len(path))
		binary.LittleEndian.PutUint16(payload[0:2], 1) // 1 segment
		binary.LittleEndian.PutUint16(payload[2:4], uint16(len(path)))
		copy(payload[4:], path)

		if err := c.sendMessage(op, payload); err != nil {
			return nil, err
		}
	} else {
		// GET/DELETE: payload is just the raw key (no length prefix)
		if err := c.sendMessage(op, key); err != nil {
			return nil, err
		}
	}

	return c.readValueResponse()
}

func (c *IPCClient) sendKeyValueOp(op OpCode, key, value []byte) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, ErrClosed
	}

	if op == OpPutPath {
		// PUT_PATH: path_count(2) + [seg_len(2) + seg]... + value
		path := string(key)
		payload := make([]byte, 2+2+len(path)+len(value))
		binary.LittleEndian.PutUint16(payload[0:2], 1) // 1 segment
		binary.LittleEndian.PutUint16(payload[2:4], uint16(len(path)))
		copy(payload[4:4+len(path)], path)
		copy(payload[4+len(path):], value)

		if err := c.sendMessage(op, payload); err != nil {
			return nil, err
		}
	} else {
		// PUT: key_len(4 LE) + key + value (value is rest of payload)
		payload := make([]byte, 4+len(key)+len(value))
		binary.LittleEndian.PutUint32(payload[0:4], uint32(len(key)))
		copy(payload[4:4+len(key)], key)
		copy(payload[4+len(key):], value)

		if err := c.sendMessage(op, payload); err != nil {
			return nil, err
		}
	}

	return c.readValueResponse()
}

func (c *IPCClient) sendTxnOp(op OpCode, txnID uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrClosed
	}

	// Transaction ops: txn_id(8 LE)
	payload := make([]byte, 8)
	binary.LittleEndian.PutUint64(payload[0:8], txnID)

	if err := c.sendMessage(op, payload); err != nil {
		return err
	}

	return c.readSimpleResponse()
}

func (c *IPCClient) readValueResponse() ([]byte, error) {
	opcode, payload, err := c.readMessage()
	if err != nil {
		return nil, err
	}

	switch opcode {
	case OpOK:
		// OK with no data
		return nil, nil
	case OpValue:
		// VALUE response: payload is the value
		// Empty payload means key doesn't exist
		if len(payload) == 0 {
			return nil, nil
		}
		return payload, nil
	case OpError:
		return nil, c.parseErrorPayload(payload)
	default:
		return nil, &ProtocolError{Message: fmt.Sprintf("unexpected opcode: %#x", opcode)}
	}
}

func (c *IPCClient) readSimpleResponse() error {
	opcode, payload, err := c.readMessage()
	if err != nil {
		return err
	}

	if opcode == OpError {
		return c.parseErrorPayload(payload)
	}

	if opcode != OpOK && opcode != OpTxnID {
		return &ProtocolError{Message: fmt.Sprintf("expected OK, got opcode %#x", opcode)}
	}

	return nil
}

func (c *IPCClient) readQueryResponse() ([]KeyValue, error) {
	opcode, payload, err := c.readMessage()
	if err != nil {
		return nil, err
	}

	if opcode == OpError {
		return nil, c.parseErrorPayload(payload)
	}

	if opcode != OpValue && opcode != OpOK {
		return nil, &ProtocolError{Message: fmt.Sprintf("unexpected query response opcode: %#x", opcode)}
	}

	if len(payload) < 4 {
		// Empty result
		return []KeyValue{}, nil
	}

	// Response format: count(4 LE) + [key_len(4 LE) + key + value_len(4 LE) + value]...
	count := binary.LittleEndian.Uint32(payload[0:4])
	results := make([]KeyValue, 0, count)
	offset := 4

	for i := uint32(0); i < count; i++ {
		if offset+4 > len(payload) {
			return nil, &ProtocolError{Message: "truncated query response"}
		}
		keyLen := binary.LittleEndian.Uint32(payload[offset : offset+4])
		offset += 4

		if offset+int(keyLen)+4 > len(payload) {
			return nil, &ProtocolError{Message: "truncated query response"}
		}
		key := make([]byte, keyLen)
		copy(key, payload[offset:offset+int(keyLen)])
		offset += int(keyLen)

		valueLen := binary.LittleEndian.Uint32(payload[offset : offset+4])
		offset += 4

		if offset+int(valueLen) > len(payload) {
			return nil, &ProtocolError{Message: "truncated query response"}
		}
		value := make([]byte, valueLen)
		copy(value, payload[offset:offset+int(valueLen)])
		offset += int(valueLen)

		results = append(results, KeyValue{Key: key, Value: value})
	}

	return results, nil
}

// readScanResponse parses the SCAN response format:
// count(4 LE) + [key_len(2 LE) + key + val_len(4 LE) + val]...
func (c *IPCClient) readScanResponse() ([]KeyValue, error) {
	opcode, payload, err := c.readMessage()
	if err != nil {
		return nil, err
	}

	if opcode == OpError {
		return nil, c.parseErrorPayload(payload)
	}

	if opcode != OpValue && opcode != OpOK {
		return nil, &ProtocolError{Message: fmt.Sprintf("unexpected scan response opcode: %#x", opcode)}
	}

	if len(payload) < 4 {
		// Empty result
		return []KeyValue{}, nil
	}

	// Response format: count(4 LE) + [key_len(2 LE) + key + val_len(4 LE) + val]...
	count := binary.LittleEndian.Uint32(payload[0:4])
	results := make([]KeyValue, 0, count)
	offset := 4

	for i := uint32(0); i < count; i++ {
		if offset+2 > len(payload) {
			return nil, &ProtocolError{Message: "truncated scan response (key_len)"}
		}
		keyLen := binary.LittleEndian.Uint16(payload[offset : offset+2])
		offset += 2

		if offset+int(keyLen)+4 > len(payload) {
			return nil, &ProtocolError{Message: "truncated scan response (key+val_len)"}
		}
		key := make([]byte, keyLen)
		copy(key, payload[offset:offset+int(keyLen)])
		offset += int(keyLen)

		valueLen := binary.LittleEndian.Uint32(payload[offset : offset+4])
		offset += 4

		if offset+int(valueLen) > len(payload) {
			return nil, &ProtocolError{Message: "truncated scan response (value)"}
		}
		value := make([]byte, valueLen)
		copy(value, payload[offset:offset+int(valueLen)])
		offset += int(valueLen)

		results = append(results, KeyValue{Key: key, Value: value})
	}

	return results, nil
}

// KeyValue represents a key-value pair returned from queries.
type KeyValue struct {
	Key   []byte
	Value []byte
}

// StorageStats contains database storage statistics.
type StorageStats struct {
	MemtableSizeBytes  uint64
	WALSizeBytes       uint64
	ActiveTransactions int
}
