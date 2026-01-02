#!/usr/bin/env python3
"""
ToonDB gRPC Server CLI - Production-grade wrapper for toondb-grpc-server

Handles:
- Port availability checking
- Graceful shutdown with signal handling
- Health checks
- Comprehensive error handling
- Platform differences
"""

import sys
import os
import signal
import socket
import subprocess
import argparse
import time
import atexit
from pathlib import Path
from typing import Optional


# ============================================================================
# Constants
# ============================================================================

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50051
STARTUP_TIMEOUT_SECONDS = 10
HEALTH_CHECK_INTERVAL = 0.1

EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_BINARY_NOT_FOUND = 2
EXIT_PORT_IN_USE = 3
EXIT_PERMISSION_DENIED = 4
EXIT_STARTUP_FAILED = 5
EXIT_INTERRUPTED = 130


# ============================================================================
# Binary Resolution
# ============================================================================

def get_grpc_server_binary() -> str:
    """
    Find the toondb-grpc-server binary with robust fallback chain.
    
    Resolution order:
    1. TOONDB_GRPC_SERVER_PATH environment variable
    2. Bundled in wheel (_bin/<platform>/)
    3. System PATH
    4. Development build (../target/release/)
    
    Returns:
        Path to the binary
        
    Raises:
        FileNotFoundError: If binary cannot be found
    """
    # 1. Environment variable override
    env_path = os.environ.get("TOONDB_GRPC_SERVER_PATH")
    if env_path:
        if os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path
        raise FileNotFoundError(
            f"TOONDB_GRPC_SERVER_PATH set to '{env_path}' but file not found or not executable"
        )
    
    # Platform-specific binary name
    binary_name = "toondb-grpc-server.exe" if sys.platform == "win32" else "toondb-grpc-server"
    
    # 2. Bundled in package
    try:
        import toondb
        package_dir = Path(toondb.__file__).parent
        bin_dir = package_dir / "_bin"
        
        if bin_dir.exists():
            for platform_dir in bin_dir.iterdir():
                if platform_dir.is_dir():
                    binary_path = platform_dir / binary_name
                    if binary_path.exists() and os.access(binary_path, os.X_OK):
                        return str(binary_path)
        
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
        f"toondb-grpc-server binary not found.\n"
        f"Searched:\n"
        f"  - TOONDB_GRPC_SERVER_PATH environment variable\n"
        f"  - Bundled in package (_bin/)\n"
        f"  - System PATH\n"
        f"  - Development build (target/release/)\n"
        f"\nTo fix:\n"
        f"  1. Reinstall: pip install --force-reinstall toondb-client\n"
        f"  2. Or build: cargo build --release -p toondb-grpc-server\n"
        f"  3. Or set: export TOONDB_GRPC_SERVER_PATH=/path/to/toondb-grpc-server"
    )


# ============================================================================
# Network Utilities
# ============================================================================

def is_port_available(host: str, port: int) -> bool:
    """Check if a TCP port is available for binding."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False


def get_process_on_port(port: int) -> Optional[str]:
    """
    Try to identify what process is using a port.
    Returns a description string or None.
    """
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        try:
            import subprocess
            result = subprocess.run(
                ["lsof", "-i", f":{port}", "-P", "-n"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    # Parse first result line
                    parts = lines[1].split()
                    if len(parts) >= 2:
                        return f"{parts[0]} (PID: {parts[1]})"
        except Exception:
            pass
    return None


def wait_for_port_ready(host: str, port: int, timeout: float = STARTUP_TIMEOUT_SECONDS) -> bool:
    """Wait for a port to accept connections."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((host, port))
            sock.close()
            return True
        except (socket.error, OSError):
            time.sleep(HEALTH_CHECK_INTERVAL)
    
    return False


# ============================================================================
# Server Process Management
# ============================================================================

class GrpcServerProcess:
    """Manages the toondb-grpc-server subprocess with proper lifecycle handling."""
    
    def __init__(self, binary: str, args: argparse.Namespace):
        self.binary = binary
        self.args = args
        self.process: Optional[subprocess.Popen] = None
        self._shutdown_requested = False
    
    def build_command(self) -> list:
        """Build the command line for the server."""
        cmd = [
            self.binary,
            "--host", self.args.host,
            "--port", str(self.args.port),
        ]
        
        if self.args.debug:
            cmd.append("--debug")
        
        return cmd
    
    def start(self) -> int:
        """Start the gRPC server process."""
        host = self.args.host
        port = self.args.port
        
        # Check port availability
        if not is_port_available(host, port):
            process_info = get_process_on_port(port)
            if process_info:
                print(
                    f"[gRPC] Error: Port {port} is already in use by {process_info}",
                    file=sys.stderr
                )
            else:
                print(
                    f"[gRPC] Error: Port {port} is already in use\n"
                    f"       Try a different port with --port <PORT>",
                    file=sys.stderr
                )
            return EXIT_PORT_IN_USE
        
        # Check if binding to privileged port
        if port < 1024 and os.geteuid() != 0:
            print(
                f"[gRPC] Error: Port {port} requires root privileges\n"
                f"       Use a port >= 1024 or run with sudo",
                file=sys.stderr
            )
            return EXIT_PERMISSION_DENIED
        
        # Build command
        cmd = self.build_command()
        
        # Register cleanup handlers
        self._register_signal_handlers()
        
        # Start process
        try:
            print(f"[gRPC] Starting toondb-grpc-server...", file=sys.stderr)
            print(f"[gRPC] Listening on {host}:{port}", file=sys.stderr)
            
            self.process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            
            # Wait for server to be ready
            if not self.args.no_wait:
                print(f"[gRPC] Waiting for server to be ready...", file=sys.stderr)
                if wait_for_port_ready(host, port):
                    print(f"[gRPC] Ready! gRPC endpoint: {host}:{port}", file=sys.stderr)
                else:
                    if self.process.poll() is not None:
                        print(f"[gRPC] Error: Server process exited during startup", file=sys.stderr)
                        return EXIT_STARTUP_FAILED
                    print(f"[gRPC] Warning: Server may not be ready yet", file=sys.stderr)
            
            # Wait for process to complete
            return_code = self.process.wait()
            return return_code
            
        except FileNotFoundError:
            print(f"[gRPC] Error: Binary not found: {self.binary}", file=sys.stderr)
            return EXIT_BINARY_NOT_FOUND
        except PermissionError:
            print(f"[gRPC] Error: Permission denied executing: {self.binary}", file=sys.stderr)
            return EXIT_PERMISSION_DENIED
        except Exception as e:
            print(f"[gRPC] Error: {e}", file=sys.stderr)
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
            print(f"\n[gRPC] Force shutdown...", file=sys.stderr)
            if self.process:
                self.process.kill()
            sys.exit(EXIT_INTERRUPTED)
        
        self._shutdown_requested = True
        signal_name = signal.Signals(signum).name
        print(f"\n[gRPC] Received {signal_name}, shutting down gracefully...", file=sys.stderr)
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                print(f"[gRPC] Shutdown complete", file=sys.stderr)
            except subprocess.TimeoutExpired:
                print(f"[gRPC] Timeout waiting for shutdown, forcing...", file=sys.stderr)
                self.process.kill()


def check_status(host: str, port: int) -> int:
    """Check if gRPC server is running."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((host, port))
        sock.close()
        print(f"[gRPC] Running on {host}:{port}")
        return EXIT_SUCCESS
    except (socket.error, OSError):
        print(f"[gRPC] Not running on {host}:{port}")
        return EXIT_GENERAL_ERROR


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for toondb-grpc-server CLI."""
    parser = argparse.ArgumentParser(
        prog="toondb-grpc-server",
        description="ToonDB gRPC Server - Remote vector search operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (default)   Start the gRPC server
  status      Check if server is running

Examples:
  # Start server on default port
  toondb-grpc-server

  # Custom host and port
  toondb-grpc-server --host 0.0.0.0 --port 50051

  # Enable debug logging
  toondb-grpc-server --debug

  # Check status
  toondb-grpc-server status --port 50051

Environment Variables:
  TOONDB_GRPC_SERVER_PATH   Path to binary (overrides bundled)

Python Usage:
  import grpc
  from toondb_pb2_grpc import VectorServiceStub
  
  channel = grpc.insecure_channel('localhost:50051')
  stub = VectorServiceStub(channel)
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.add_argument("--host", default=DEFAULT_HOST, help="Server host")
    status_parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help="Server port")
    
    # Main arguments
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Listen port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for server to be ready"
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
            print(f"toondb-grpc-server (Python wrapper) {__version__}")
        except ImportError:
            print("toondb-grpc-server (Python wrapper)")
        return EXIT_SUCCESS
    
    # Handle status command
    if args.command == "status":
        return check_status(args.host, args.port)
    
    # Start server
    try:
        binary = get_grpc_server_binary()
    except FileNotFoundError as e:
        print(f"[gRPC] Error: {e}", file=sys.stderr)
        return EXIT_BINARY_NOT_FOUND
    
    server = GrpcServerProcess(binary, args)
    return server.start()


if __name__ == "__main__":
    sys.exit(main())
