#!/usr/bin/env node

/**
 * ToonDB Bulk CLI Wrapper
 * Automatically locates and runs the platform-specific binary.
 */

const os = require('os');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

function getPlatformBinary() {
    const platform = os.platform();
    const arch = os.arch();

    let target = '';
    let ext = '';

    if (platform === 'darwin') {
        target = arch === 'arm64' ? 'aarch64-apple-darwin' : 'x86_64-apple-darwin';
    } else if (platform === 'linux') {
        target = arch === 'arm64' ? 'aarch64-unknown-linux-gnu' : 'x86_64-unknown-linux-gnu';
    } else if (platform === 'win32') {
        target = 'x86_64-pc-windows-msvc';
        ext = '.exe';
    } else {
        throw new Error(`Unsupported platform: ${platform} ${arch}`);
    }

    // Look for bundled binary
    const binName = `toondb-bulk${ext}`;
    const bundledPath = path.resolve(__dirname, '..', '_bin', target, binName);

    if (fs.existsSync(bundledPath)) {
        return bundledPath;
    }

    // Fallback: Check environment variable
    if (process.env.TOONDB_BULK_PATH) {
        return process.env.TOONDB_BULK_PATH;
    }

    throw new Error(`
toondb-bulk binary not found!
Searched at: ${bundledPath}

To fix:
1. Reinstall the package: npm install --force @sushanth/toondb
2. Or set TOONDB_BULK_PATH environment variable
    `);
}

try {
    const binPath = getPlatformBinary();

    const child = spawn(binPath, process.argv.slice(2), {
        stdio: 'inherit'
    });

    child.on('exit', (code) => {
        process.exit(code || 0);
    });

    child.on('error', (err) => {
        console.error(`Failed to start toondb-bulk: ${err.message}`);
        process.exit(1);
    });

    const signals = ['SIGINT', 'SIGTERM'];
    signals.forEach(signal => {
        process.on(signal, () => {
            child.kill(signal);
        });
    });

} catch (err) {
    console.error(`[toondb-js] Error: ${err.message}`);
    process.exit(1);
}
