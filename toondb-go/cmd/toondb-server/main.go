package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

func findBinary(name string) (string, error) {
	// 1. Check environment variable
	if envPath := os.Getenv("TOONDB_SERVER_PATH"); envPath != "" {
		if _, err := os.Stat(envPath); err == nil {
			return envPath, nil
		}
	}

	// 2. Check current directory (dev/testing)
	cwd, _ := os.Getwd()
	localPath := filepath.Join(cwd, name)
	if _, err := os.Stat(localPath); err == nil {
		return localPath, nil
	}

	// 3. Check standard PATH
	path, err := exec.LookPath(name)
	if err == nil {
		return path, nil
	}

	// 4. Check common installation paths
	homeDir, _ := os.UserHomeDir()
	commonPaths := []string{
		filepath.Join(homeDir, ".toondb", "bin", name),
		"/usr/local/bin/" + name,
		"/opt/toondb/bin/" + name,
	}

	for _, p := range commonPaths {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	return "", fmt.Errorf("binary not found")
}

func main() {
	binaryName := "toondb-server"
	if runtime.GOOS == "windows" {
		binaryName += ".exe"
	}

	binPath, err := findBinary(binaryName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s binary not found.\n", binaryName)
		fmt.Fprintln(os.Stderr, "Please ensure ToonDB is installed or set TOONDB_SERVER_PATH.")
		fmt.Fprintln(os.Stderr, "Download from: https://github.com/sushanth/toondb/releases")
		os.Exit(1)
	}

	// EXEC the binary, replacing current process
	// Note: syscall.Exec is Unix-specific. For cross-platform, we use os/exec and wait.
	cmd := exec.Command(binPath, os.Args[1:]...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Handle signals? 
	// Go's exec.Command doesn't automatically forward signals, but it waits.
	// For a simple wrapper, this is usually sufficient, but for long running processes 
	// like a server, we might want to handle it.
	// However, for simplicity here, we rely on the user sending signals to the process group.

	err = cmd.Run()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "Error executing %s: %v\n", binaryName, err)
		os.Exit(1)
	}
}
