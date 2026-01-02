package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

func findBinary(name string) (string, error) {
	if envPath := os.Getenv("TOONDB_BULK_PATH"); envPath != "" {
		if _, err := os.Stat(envPath); err == nil {
			return envPath, nil
		}
	}
	cwd, _ := os.Getwd()
	localPath := filepath.Join(cwd, name)
	if _, err := os.Stat(localPath); err == nil {
		return localPath, nil
	}
	path, err := exec.LookPath(name)
	if err == nil {
		return path, nil
	}
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
	binaryName := "toondb-bulk"
	if runtime.GOOS == "windows" {
		binaryName += ".exe"
	}

	binPath, err := findBinary(binaryName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s binary not found.\n", binaryName)
		fmt.Fprintln(os.Stderr, "Please ensure ToonDB is installed or set TOONDB_BULK_PATH.")
		os.Exit(1)
	}

	cmd := exec.Command(binPath, os.Args[1:]...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		os.Exit(1)
	}
}
