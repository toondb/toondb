/**
 * ToonDB Embedded Server Manager
 *
 * Manages the lifecycle of the ToonDB server process for embedded mode.
 * Automatically starts the server when needed and stops it on cleanup.
 *
 * @packageDocumentation
 */

// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import * as fs from 'fs';
import * as path from 'path';
import * as net from 'net';
import { spawn, ChildProcess } from 'child_process';
import { ConnectionError, DatabaseError } from './errors';

/**
 * Find the toondb-server binary (provides IPC interface)
 */
function findServerBinary(): string {
  const platform = process.platform;
  const arch = process.arch;

  let target: string;
  if (platform === 'darwin') {
    target = arch === 'arm64' ? 'aarch64-apple-darwin' : 'x86_64-apple-darwin';
  } else if (platform === 'win32') {
    target = 'x86_64-pc-windows-msvc';
  } else {
    target = arch === 'arm64' ? 'aarch64-unknown-linux-gnu' : 'x86_64-unknown-linux-gnu';
  }

  const binaryName = platform === 'win32' ? 'toondb-server.exe' : 'toondb-server';

  // Search paths - prioritize bundled binaries
  const searchPaths = [
    // Bundled in package (installed via npm) - from dist/cjs or dist/esm
    path.join(__dirname, '..', '_bin', target, binaryName),
    path.join(__dirname, '..', '..', '_bin', target, binaryName),
    path.join(__dirname, '..', '..', '..', '_bin', target, binaryName),
    // When running from source (src/) during development/testing
    path.resolve(__dirname, '..', '_bin', target, binaryName),
    // Development paths - from project root
    path.join(__dirname, '..', '..', 'target', 'release', binaryName),
    path.join(__dirname, '..', '..', 'target', 'debug', binaryName),
    path.join(__dirname, '..', '..', '..', 'target', 'release', binaryName),
    path.join(__dirname, '..', '..', '..', 'target', 'debug', binaryName),
    // Absolute paths for toondb workspace
    path.resolve(process.cwd(), '_bin', target, binaryName),
    path.resolve(process.cwd(), 'target', 'release', binaryName),
    path.resolve(process.cwd(), '..', 'target', 'release', binaryName),
  ];

  for (const p of searchPaths) {
    if (fs.existsSync(p)) {
      return p;
    }
  }

  // Try PATH
  const pathDirs = (process.env.PATH || '').split(path.delimiter);
  for (const dir of pathDirs) {
    const p = path.join(dir, binaryName);
    if (fs.existsSync(p)) {
      return p;
    }
  }

  throw new DatabaseError(
    `Could not find ${binaryName}. ` +
    `The pre-built binary may not be available for your platform (${platform}/${arch}). ` +
    `Install via: cargo build --release -p toondb-tools`
  );
}

/**
 * Wait for a Unix socket to become available
 */
async function waitForSocket(socketPath: string, timeoutMs: number = 10000): Promise<void> {
  const startTime = Date.now();
  const checkInterval = 100;

  while (Date.now() - startTime < timeoutMs) {
    if (fs.existsSync(socketPath)) {
      // Try to connect to verify it's actually listening
      try {
        await new Promise<void>((resolve, reject) => {
          const socket = net.createConnection({ path: socketPath }, () => {
            socket.destroy();
            resolve();
          });
          socket.on('error', reject);
          socket.setTimeout(1000);
        });
        return;
      } catch {
        // Socket exists but not ready yet
      }
    }
    await new Promise(resolve => setTimeout(resolve, checkInterval));
  }

  throw new ConnectionError(
    `Timeout waiting for server socket at ${socketPath} after ${timeoutMs}ms`
  );
}

/**
 * Embedded server instance
 */
interface ServerInstance {
  process: ChildProcess;
  socketPath: string;
  dbPath: string;
}

// Track running server instances
const runningServers: Map<string, ServerInstance> = new Map();

/**
 * Start an embedded ToonDB server for the given database path.
 * If a server is already running for this path, return the existing instance.
 *
 * @param dbPath - Path to the database directory
 * @returns The socket path for connecting
 */
export async function startEmbeddedServer(dbPath: string): Promise<string> {
  const absolutePath = path.resolve(dbPath);
  const socketPath = path.join(absolutePath, 'toondb.sock');

  // Check if server already running for this path
  const existing = runningServers.get(absolutePath);
  if (existing && existing.process.exitCode === null) {
    // Server still running
    return socketPath;
  }

  // Check if another process already has a server running
  if (fs.existsSync(socketPath)) {
    try {
      // Try to connect - if successful, server is running
      await waitForSocket(socketPath, 1000);
      return socketPath;
    } catch {
      // Socket exists but dead - clean it up
      try {
        fs.unlinkSync(socketPath);
      } catch {
        // Ignore
      }
    }
  }

  // Ensure database directory exists
  if (!fs.existsSync(absolutePath)) {
    fs.mkdirSync(absolutePath, { recursive: true });
  }

  // Find and spawn server
  const serverBinary = findServerBinary();
  
  const serverProcess = spawn(serverBinary, ['--db', absolutePath, '--socket', socketPath], {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
  });

  // Collect stderr for error reporting
  let stderrOutput = '';
  serverProcess.stderr?.on('data', (data) => {
    stderrOutput += data.toString();
  });

  // Handle process exit
  serverProcess.on('exit', (code, signal) => {
    runningServers.delete(absolutePath);
    if (code !== 0 && code !== null) {
      console.error(`ToonDB server exited with code ${code}: ${stderrOutput}`);
    }
  });

  serverProcess.on('error', (err) => {
    runningServers.delete(absolutePath);
    console.error(`Failed to start ToonDB server: ${err.message}`);
  });

  // Store instance
  runningServers.set(absolutePath, {
    process: serverProcess,
    socketPath,
    dbPath: absolutePath,
  });

  // Wait for server to be ready
  try {
    await waitForSocket(socketPath, 10000);
  } catch (err) {
    // Kill the process if it didn't start properly
    serverProcess.kill();
    runningServers.delete(absolutePath);
    throw new DatabaseError(
      `Failed to start embedded server: ${stderrOutput || (err as Error).message}`
    );
  }

  return socketPath;
}

/**
 * Stop the embedded server for a specific database path
 */
export async function stopEmbeddedServer(dbPath: string): Promise<void> {
  const absolutePath = path.resolve(dbPath);
  const instance = runningServers.get(absolutePath);

  if (!instance) {
    return;
  }

  // Send SIGTERM and wait for graceful shutdown
  instance.process.kill('SIGTERM');

  // Wait for process to exit (max 5 seconds)
  await new Promise<void>((resolve) => {
    const timeout = setTimeout(() => {
      // Force kill if still running
      if (instance.process.exitCode === null) {
        instance.process.kill('SIGKILL');
      }
      resolve();
    }, 5000);

    instance.process.on('exit', () => {
      clearTimeout(timeout);
      resolve();
    });
  });

  runningServers.delete(absolutePath);

  // Clean up socket file
  try {
    if (fs.existsSync(instance.socketPath)) {
      fs.unlinkSync(instance.socketPath);
    }
  } catch {
    // Ignore cleanup errors
  }
}

/**
 * Stop all running embedded servers
 */
export async function stopAllEmbeddedServers(): Promise<void> {
  const stopPromises: Promise<void>[] = [];
  for (const [dbPath] of runningServers) {
    stopPromises.push(stopEmbeddedServer(dbPath));
  }
  await Promise.all(stopPromises);
}

/**
 * Check if an embedded server is running for a database path
 */
export function isServerRunning(dbPath: string): boolean {
  const absolutePath = path.resolve(dbPath);
  const instance = runningServers.get(absolutePath);
  return instance !== undefined && instance.process.exitCode === null;
}

// Cleanup on process exit
process.on('exit', () => {
  for (const instance of runningServers.values()) {
    try {
      instance.process.kill('SIGKILL');
    } catch {
      // Ignore
    }
  }
});

process.on('SIGINT', async () => {
  await stopAllEmbeddedServers();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await stopAllEmbeddedServers();
  process.exit(0);
});
