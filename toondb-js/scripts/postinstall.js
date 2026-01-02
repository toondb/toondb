#!/usr/bin/env node
/**
 * Post-install script to ensure native binaries are executable on Unix systems.
 * This is required because npm doesn't preserve executable permissions during publish.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Only run on Unix-like systems
if (process.platform === 'win32') {
  console.log('[toondb] Skipping chmod on Windows');
  process.exit(0);
}

const binDir = path.join(__dirname, '..', '_bin');

if (!fs.existsSync(binDir)) {
  console.log('[toondb] No _bin directory found, skipping postinstall (will be populated on first use)');
  process.exit(0);
}

const binaryNames = ['toondb-server', 'toondb-bulk', 'toondb-grpc-server'];
const targets = [
  'aarch64-apple-darwin',
  'x86_64-apple-darwin',
  'aarch64-unknown-linux-gnu',
  'x86_64-unknown-linux-gnu'
];

let fixed = 0;
for (const target of targets) {
  for (const binaryName of binaryNames) {
    const binaryPath = path.join(binDir, target, binaryName);
    if (fs.existsSync(binaryPath)) {
      try {
        // Make executable
        fs.chmodSync(binaryPath, 0o755);
        fixed++;
      } catch (err) {
        console.warn(`[toondb] Warning: Could not chmod ${binaryPath}: ${err.message}`);
      }
    }
  }
}

if (fixed > 0) {
  console.log(`[toondb] Made ${fixed} native binaries executable`);
}
