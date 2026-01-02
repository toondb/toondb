#!/usr/bin/env python3
"""
ToonDB Server CLI - Production-grade wrapper for toondb-server

Handles:
- Signal handling (graceful shutdown)
- Socket file cleanup (stale socket detection)
- Health checks (wait for server ready)
- Platform differences (Unix vs Windows)
- Process management (PID tracking, status checks)
- Comprehensive error handling
"""

import sys
import os
import signal
import socket
import subprocess
import argparse
import time
import atexit
import tempfile
import stat
from pathlib import Path
from typing import Optional, Tuple
from contextlib import contextmanager


# ============================================================================
# Constants
# ============================================================================

DEFAULT_DB_PATH = "./toondb_data"
DEFAULT_MAX_CLIENTS = 100
DEFAULT_TIMEOUT_MS = 30000
DEFAULT_LOG_LEVEL = "info"
STARTUP_TIMEOUT_SECONDS = 10
HEALTH_CHECK_INTERVAL = 0.1
PID_FILE_SUFFIX = ".pid"

# Exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_BINARY_NOT_FOUND = 2
EXIT_SOCKET_IN_USE = 3
EXIT_PERMISSION_DENIED = 4
EXIT_STARTUP_FAILED = 5
EXIT_INTERRUPTED = 130


# ============================================================================
# Binary Resolution
# ============================================================================

def get_server_binary() -> str:
    """
    Find the toondb-server binary with robust fallback chain.
    
    Resolution order:
    1. TOONDB_SERVER_PATH environment variable
    2. Bundled in wheel (_bin/<platform>/)
    3. System PATH
    4. Development build (../target/release/)
    
    Returns:
        Path to the binary
        
    Raises:
        FileNotFoundError: If binary cannot be found
    """
    # 1. Environment variable override
    env_path = os.environ.get("TOONDB_SERVER_PATH")
    if env_path:
        if os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path
        raise FileNotFoundError(
            f"TOONDB_SERVER_PATH set to '{env_path}' but file not found or not executable"
        )
    
    # Platform-specific binary name
    binary_name = "toondb-server.exe" if sys.platform == "win32" else "toondb-server"
    
    # 2. Bundled in package
    try:
        import toondb
        package_dir = Path(toondb.__file__).parent
        bin_dir = package_dir / "_bin"
        
        if bin_dir.exists():
            # Try platform-specific subdirectories
            for platform_dir in bin_dir.iterdir():
                if platform_dir.is_dir():
                    binary_path = platform_dir / binary_name
                    if binary_path.exists() and os.access(binary_path, os.X_OK):
                        return str(binary_path)
        
        # Try root package directory
        binary_path = package_dir / binary_name
        if binary_path.exists() and os.access(binary_path, os.X_OK):
            return str(binary_path)
    except ImportError:
        pass
    
    # 3. System PATH
    from shutil import which
    system_binary = which(binary_name)
    if system_binary:
        return system_binary
    
    # 4. Development build
    dev_paths = [
        Path(__file__).parent.parent.parent.parent / "target" / "release" / binary_name,
        Path(__file__).parent.parent.parent.parent / "target" / "debug" / binary_name,
        Path.cwd() / "target" / "release" / binary_name,
    ]
    for dev_path in dev_paths:
        if dev_path.exists() and os.access(dev_path, os.X_OK):
            return str(dev_path)
    
    raise FileNotFoundError(
        f"toondb-server binary not found.\n"
        f"Searched:\n"
        f"  - TOONDB_SERVER_PATH environment variable\n"
        f"  - Bundled in package (_bin/)\n"
        f"  - System PATH\n"
        f"  - Development build (target/release/)\n"
        f"\nTo fix:\n"
        f"  1. Reinstall: pip install --force-reinstall toondb-client\n"
        f"  2. Or build: cargo build --release -p toondb-server\n"
        f"  3. Or set: export TOONDB_SERVER_PATH=/path/to/toondb-server"
    )


# ============================================================================
# Socket Management
# ============================================================================

def get_socket_path(db_path: str, custom_socket: Optional[str] = None) -> Path:
    """Get the Unix socket path for the database."""
    if custom_socket:
        return Path(custom_socket)
    return Path(db_path) / "toondb.sock"


def get_pid_file_path(socket_path: Path) -> Path:
    """Get the PID file path for a given socket."""
    return socket_path.with_suffix(PID_FILE_SUFFIX)


def is_socket_in_use(socket_path: Path) -> Tuple[bool, Optional[int]]:
    """
    Check if a Unix socket is actively in use.
    
    Returns:
        (in_use: bool, pid: Optional[int]) - Whether socket is in use and by which PID
    """
    if not socket_path.exists():
        return False, None
    
    # Check if it's actually a socket
    try:
        mode = socket_path.stat().st_mode
        if not stat.S_ISSOCK(mode):
            return False, None
    except OSError:
        return False, None
    
    # Try to connect to see if server is running
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(str(socket_path))
        sock.close()
        
        # Socket is active, try to get PID
        pid_file = get_pid_file_path(socket_path)
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                # Verify process exists
                os.kill(pid, 0)
                return True, pid
            except (ValueError, OSError):
                pass
        return True, None
    except (socket.error, OSError):
        # Socket exists but not accepting connections - stale
        return False, None


def cleanup_stale_socket(socket_path: Path) -> bool:
    """
    Remove a stale socket file.
    
    Returns:
        True if cleanup was performed, False if socket was active
    """
    in_use, pid = is_socket_in_use(socket_path)
    
    if in_use:
        return False
    
    # Remove stale socket
    try:
        if socket_path.exists():
            socket_path.unlink()
            print(f"[Server] Cleaned up stale socket: {socket_path}", file=sys.stderr)
        
        # Also clean up PID file
        pid_file = get_pid_file_path(socket_path)
        if pid_file.exists():
            pid_file.unlink()
        
        return True
    except OSError as e:
        print(f"[Server] Warning: Could not clean up stale socket: {e}", file=sys.stderr)
        return True


def write_pid_file(socket_path: Path, pid: int) -> None:
    """Write PID file for the server."""
    pid_file = get_pid_file_path(socket_path)
    try:
        pid_file.write_text(str(pid))
    except OSError as e:
        print(f"[Server] Warning: Could not write PID file: {e}", file=sys.stderr)


def remove_pid_file(socket_path: Path) -> None:
    """Remove PID file on shutdown."""
    pid_file = get_pid_file_path(socket_path)
    try:
        if pid_file.exists():
            pid_file.unlink()
    except OSError:
        pass


# ============================================================================
# Health Checks
# ============================================================================

def wait_for_server_ready(socket_path: Path, timeout: float = STARTUP_TIMEOUT_SECONDS) -> bool:
    """
    Wait for the server to be ready to accept connections.
    
    Args:
        socket_path: Path to the Unix socket
        timeout: Maximum seconds to wait
        
    Returns:
        True if server is ready, False if timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(str(socket_path))
            sock.close()
            return True
        except (socket.error, OSError):
            time.sleep(HEALTH_CHECK_INTERVAL)
    
    return False


def verify_database_directory(db_path: str) -> None:
    """
    Verify and create database directory if needed.
    
    Raises:
        PermissionError: If directory is not writable
        OSError: If directory cannot be created
    """
    db_dir = Path(db_path)
    
    # Create if doesn't exist
    if not db_dir.exists():
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Server] Created database directory: {db_dir}", file=sys.stderr)
        except PermissionError:
            raise PermissionError(f"Cannot create database directory: {db_dir}")
    
    # Verify writable
    if not os.access(db_dir, os.W_OK):
        raise PermissionError(f"Database directory is not writable: {db_dir}")


# ============================================================================
# Process Management
# ============================================================================

class ServerProcess:
    """Manages the toondb-server subprocess with proper lifecycle handling."""
    
    def __init__(self, binary: str, args: argparse.Namespace):
        self.binary = binary
        self.args = args
        self.process: Optional[subprocess.Popen] = None
        self.socket_path = get_socket_path(args.db, args.socket)
        self._shutdown_requested = False
    
    def build_command(self) -> list:
        """Build the command line for the server."""
        cmd = [
            self.binary,
            "--db", self.args.db,
        ]
        
        if self.args.socket:
            cmd.extend(["--socket", self.args.socket])
        
        cmd.extend(["--max-clients", str(self.args.max_clients)])
        cmd.extend(["--timeout-ms", str(self.args.timeout_ms)])
        cmd.extend(["--log-level", self.args.log_level])
        
        return cmd
    
    def start(self) -> int:
        """
        Start the server process.
        
        Returns:
            Exit code
        """
        # Verify database directory
        try:
            verify_database_directory(self.args.db)
        except (PermissionError, OSError) as e:
            print(f"[Server] Error: {e}", file=sys.stderr)
            return EXIT_PERMISSION_DENIED
        
        # Check for existing socket
        in_use, existing_pid = is_socket_in_use(self.socket_path)
        if in_use:
            pid_info = f" (PID: {existing_pid})" if existing_pid else ""
            print(
                f"[Server] Error: Socket already in use{pid_info}: {self.socket_path}\n"
                f"         Another toondb-server instance may be running.\n"
                f"         Use 'toondb-server stop --socket {self.socket_path}' to stop it.",
                file=sys.stderr
            )
            return EXIT_SOCKET_IN_USE
        
        # Clean up stale socket
        cleanup_stale_socket(self.socket_path)
        
        # Build command
        cmd = self.build_command()
        
        # Register cleanup handlers
        self._register_signal_handlers()
        atexit.register(self._cleanup)
        
        # Start process
        try:
            print(f"[Server] Starting toondb-server...", file=sys.stderr)
            print(f"[Server] Database: {Path(self.args.db).absolute()}", file=sys.stderr)
            print(f"[Server] Socket: {self.socket_path}", file=sys.stderr)
            
            self.process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            
            # Write PID file
            write_pid_file(self.socket_path, self.process.pid)
            
            # Wait for server to be ready
            if not self.args.no_wait:
                print(f"[Server] Waiting for server to be ready...", file=sys.stderr)
                if wait_for_server_ready(self.socket_path):
                    print(f"[Server] Ready! Accepting connections on {self.socket_path}", file=sys.stderr)
                else:
                    # Check if process died
                    if self.process.poll() is not None:
                        print(f"[Server] Error: Server process exited during startup", file=sys.stderr)
                        return EXIT_STARTUP_FAILED
                    print(f"[Server] Warning: Server may not be ready yet", file=sys.stderr)
            
            # Wait for process to complete
            return_code = self.process.wait()
            return return_code
            
        except FileNotFoundError:
            print(f"[Server] Error: Binary not found: {self.binary}", file=sys.stderr)
            return EXIT_BINARY_NOT_FOUND
        except PermissionError:
            print(f"[Server] Error: Permission denied executing: {self.binary}", file=sys.stderr)
            return EXIT_PERMISSION_DENIED
        except Exception as e:
            print(f"[Server] Error: {e}", file=sys.stderr)
            return EXIT_GENERAL_ERROR
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGHUP, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        if self._shutdown_requested:
            # Force kill on second signal
            print(f"\n[Server] Force shutdown...", file=sys.stderr)
            if self.process:
                self.process.kill()
            sys.exit(EXIT_INTERRUPTED)
        
        self._shutdown_requested = True
        signal_name = signal.Signals(signum).name
        print(f"\n[Server] Received {signal_name}, shutting down gracefully...", file=sys.stderr)
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[Server] Timeout waiting for shutdown, forcing...", file=sys.stderr)
                self.process.kill()
    
    def _cleanup(self) -> None:
        """Cleanup on exit."""
        remove_pid_file(self.socket_path)
        
        # Try to remove socket file (server should have done this)
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except OSError:
            pass


def stop_server(socket_path_str: Optional[str], db_path: str) -> int:
    """Stop a running server by socket path."""
    socket_path = get_socket_path(db_path, socket_path_str)
    
    in_use, pid = is_socket_in_use(socket_path)
    
    if not in_use:
        print(f"[Server] No server running on {socket_path}", file=sys.stderr)
        return EXIT_SUCCESS
    
    if pid:
        print(f"[Server] Stopping server (PID: {pid})...", file=sys.stderr)
        try:
            os.kill(pid, signal.SIGTERM)
            
            # Wait for shutdown
            for _ in range(50):  # 5 seconds
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except OSError:
                    break
            else:
                # Force kill
                print(f"[Server] Force killing server...", file=sys.stderr)
                os.kill(pid, signal.SIGKILL)
            
            print(f"[Server] Stopped", file=sys.stderr)
            cleanup_stale_socket(socket_path)
            return EXIT_SUCCESS
        except ProcessLookupError:
            print(f"[Server] Process not found, cleaning up...", file=sys.stderr)
            cleanup_stale_socket(socket_path)
            return EXIT_SUCCESS
        except PermissionError:
            print(f"[Server] Error: Permission denied to stop process {pid}", file=sys.stderr)
            return EXIT_PERMISSION_DENIED
    else:
        print(f"[Server] Server is running but PID unknown. Remove socket manually: {socket_path}", file=sys.stderr)
        return EXIT_GENERAL_ERROR


def server_status(socket_path_str: Optional[str], db_path: str) -> int:
    """Check server status."""
    socket_path = get_socket_path(db_path, socket_path_str)
    
    in_use, pid = is_socket_in_use(socket_path)
    
    if in_use:
        pid_info = f" (PID: {pid})" if pid else ""
        print(f"[Server] Running{pid_info}")
        print(f"         Socket: {socket_path}")
        print(f"         Database: {Path(db_path).absolute()}")
        return EXIT_SUCCESS
    else:
        print(f"[Server] Not running")
        if socket_path.exists():
            print(f"         Stale socket: {socket_path}")
        return EXIT_GENERAL_ERROR


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for toondb-server CLI."""
    parser = argparse.ArgumentParser(
        prog="toondb-server",
        description="ToonDB IPC Server - Multi-process database access via Unix sockets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (default)   Start the server
  stop        Stop a running server
  status      Check server status

Examples:
  # Start server
  toondb-server --db ./my_database

  # Start with custom settings
  toondb-server --db ./prod_db --max-clients 200 --log-level info

  # Check status
  toondb-server status --db ./my_database

  # Stop server
  toondb-server stop --db ./my_database

Environment Variables:
  TOONDB_SERVER_PATH   Path to toondb-server binary (overrides bundled)
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop a running server")
    stop_parser.add_argument("-d", "--db", default=DEFAULT_DB_PATH, help="Database directory")
    stop_parser.add_argument("-s", "--socket", help="Unix socket path")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.add_argument("-d", "--db", default=DEFAULT_DB_PATH, help="Database directory")
    status_parser.add_argument("-s", "--socket", help="Unix socket path")
    
    # Main (start) arguments
    parser.add_argument(
        "-d", "--db",
        default=DEFAULT_DB_PATH,
        help=f"Database directory (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "-s", "--socket",
        help="Unix socket path (default: <db>/toondb.sock)"
    )
    parser.add_argument(
        "--max-clients",
        type=int,
        default=DEFAULT_MAX_CLIENTS,
        help=f"Maximum concurrent connections (default: {DEFAULT_MAX_CLIENTS})"
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help=f"Connection timeout in milliseconds (default: {DEFAULT_TIMEOUT_MS})"
    )
    parser.add_argument(
        "--log-level",
        choices=["trace", "debug", "info", "warn", "error"],
        default=DEFAULT_LOG_LEVEL,
        help=f"Log level (default: {DEFAULT_LOG_LEVEL})"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for server to be ready before returning"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit"
    )
    
    args = parser.parse_args()
    
    # Version
    if args.version:
        try:
            from toondb import __version__
            print(f"toondb-server (Python wrapper) {__version__}")
        except ImportError:
            print("toondb-server (Python wrapper)")
        return EXIT_SUCCESS
    
    # Handle subcommands
    if args.command == "stop":
        return stop_server(args.socket, args.db)
    elif args.command == "status":
        return server_status(args.socket, args.db)
    
    # Start server
    try:
        binary = get_server_binary()
    except FileNotFoundError as e:
        print(f"[Server] Error: {e}", file=sys.stderr)
        return EXIT_BINARY_NOT_FOUND
    
    server = ServerProcess(binary, args)
    return server.start()


if __name__ == "__main__":
    sys.exit(main())
