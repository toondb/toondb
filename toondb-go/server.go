// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package toondb

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"time"
)

// ServerInstance tracks a running embedded server.
type ServerInstance struct {
	Process    *exec.Cmd
	SocketPath string
	DBPath     string
}

var (
	runningServers = make(map[string]*ServerInstance)
	serverMu       sync.Mutex
)

// findServerBinary locates the toondb-server binary.
func findServerBinary() (string, error) {
	goos := runtime.GOOS
	goarch := runtime.GOARCH

	// Windows doesn't support Unix sockets
	if goos == "windows" {
		return "", fmt.Errorf(
			"toondb-server is not available on Windows (requires Unix domain sockets). " +
				"Use the gRPC client for cross-platform support",
		)
	}

	// Determine target triple
	var target string
	switch goos {
	case "darwin":
		if goarch == "arm64" {
			target = "aarch64-apple-darwin"
		} else {
			target = "x86_64-apple-darwin"
		}
	case "linux":
		if goarch == "arm64" {
			target = "aarch64-unknown-linux-gnu"
		} else {
			target = "x86_64-unknown-linux-gnu"
		}
	default:
		return "", fmt.Errorf("unsupported platform: %s/%s", goos, goarch)
	}

	binaryName := "toondb-server"

	// Search paths
	cwd, _ := os.Getwd()
	searchPaths := []string{
		// Current working directory
		filepath.Join(cwd, "_bin", target, binaryName),
		filepath.Join(cwd, "target", "release", binaryName),
		filepath.Join(cwd, "target", "debug", binaryName),
		// Parent directory (if running from examples or tests)
		filepath.Join(cwd, "..", "_bin", target, binaryName),
		filepath.Join(cwd, "..", "target", "release", binaryName),
		// Toondb workspace root
		filepath.Join(cwd, "..", "..", "_bin", target, binaryName),
		filepath.Join(cwd, "..", "..", "target", "release", binaryName),
	}

	for _, p := range searchPaths {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	// Try PATH
	if path, err := exec.LookPath(binaryName); err == nil {
		return path, nil
	}

	return "", fmt.Errorf(
		"could not find %s. Install via: cargo build --release -p toondb-tools",
		binaryName,
	)
}

// waitForSocket waits for the Unix socket to become available.
func waitForSocket(socketPath string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	checkInterval := 100 * time.Millisecond

	for time.Now().Before(deadline) {
		if _, err := os.Stat(socketPath); err == nil {
			// Socket file exists, try to connect
			conn, err := net.DialTimeout("unix", socketPath, time.Second)
			if err == nil {
				conn.Close()
				return nil
			}
		}
		time.Sleep(checkInterval)
	}

	return fmt.Errorf("timeout waiting for server socket at %s after %v", socketPath, timeout)
}

// StartEmbeddedServer starts an embedded ToonDB server for the given database path.
// If a server is already running for this path, returns the existing socket path.
func StartEmbeddedServer(dbPath string) (string, error) {
	absolutePath, err := filepath.Abs(dbPath)
	if err != nil {
		return "", fmt.Errorf("failed to resolve path: %w", err)
	}

	socketPath := filepath.Join(absolutePath, "toondb.sock")

	serverMu.Lock()
	defer serverMu.Unlock()

	// Check if server already running
	if existing, ok := runningServers[absolutePath]; ok {
		if existing.Process != nil && existing.Process.Process != nil {
			// Check if process is still running
			if err := existing.Process.Process.Signal(os.Signal(nil)); err == nil {
				return socketPath, nil
			}
		}
		// Server not running, remove stale entry
		delete(runningServers, absolutePath)
	}

	// Check if socket exists (external server running)
	if _, err := os.Stat(socketPath); err == nil {
		conn, err := net.DialTimeout("unix", socketPath, time.Second)
		if err == nil {
			conn.Close()
			// External server running, use it
			return socketPath, nil
		}
		// Stale socket file, remove it
		os.Remove(socketPath)
	}

	// Find the server binary
	binaryPath, err := findServerBinary()
	if err != nil {
		return "", err
	}

	// Ensure database directory exists
	if err := os.MkdirAll(absolutePath, 0755); err != nil {
		return "", fmt.Errorf("failed to create database directory: %w", err)
	}

	// Start the server process
	cmd := exec.Command(binaryPath, "--db", absolutePath)
	cmd.Dir = absolutePath
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Start the process
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("failed to start embedded server: %w", err)
	}

	// Track the server instance
	runningServers[absolutePath] = &ServerInstance{
		Process:    cmd,
		SocketPath: socketPath,
		DBPath:     absolutePath,
	}

	// Wait for socket to be ready
	if err := waitForSocket(socketPath, 10*time.Second); err != nil {
		// Kill the process if socket never became available
		cmd.Process.Kill()
		delete(runningServers, absolutePath)
		return "", err
	}

	return socketPath, nil
}

// StopEmbeddedServer stops the embedded server for the given database path.
func StopEmbeddedServer(dbPath string) error {
	absolutePath, err := filepath.Abs(dbPath)
	if err != nil {
		return fmt.Errorf("failed to resolve path: %w", err)
	}

	serverMu.Lock()
	defer serverMu.Unlock()

	instance, ok := runningServers[absolutePath]
	if !ok {
		return nil // Not running
	}

	delete(runningServers, absolutePath)

	if instance.Process != nil && instance.Process.Process != nil {
		// Send interrupt signal for graceful shutdown
		if err := instance.Process.Process.Signal(os.Interrupt); err != nil {
			// Force kill if interrupt fails
			instance.Process.Process.Kill()
		}

		// Wait for process to exit (with timeout)
		done := make(chan error, 1)
		go func() {
			done <- instance.Process.Wait()
		}()

		select {
		case <-done:
			// Process exited
		case <-time.After(5 * time.Second):
			// Force kill after timeout
			instance.Process.Process.Kill()
		}
	}

	// Clean up socket file
	socketPath := filepath.Join(absolutePath, "toondb.sock")
	os.Remove(socketPath)

	return nil
}

// StopAllEmbeddedServers stops all running embedded servers.
func StopAllEmbeddedServers() {
	serverMu.Lock()
	paths := make([]string, 0, len(runningServers))
	for path := range runningServers {
		paths = append(paths, path)
	}
	serverMu.Unlock()

	for _, path := range paths {
		StopEmbeddedServer(path)
	}
}

// IsServerRunning checks if an embedded server is running for the given path.
func IsServerRunning(dbPath string) bool {
	absolutePath, err := filepath.Abs(dbPath)
	if err != nil {
		return false
	}

	serverMu.Lock()
	defer serverMu.Unlock()

	instance, ok := runningServers[absolutePath]
	if !ok {
		return false
	}

	if instance.Process == nil || instance.Process.Process == nil {
		return false
	}

	// Check if process is still running
	err = instance.Process.Process.Signal(os.Signal(nil))
	return err == nil
}
