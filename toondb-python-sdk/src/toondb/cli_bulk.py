#!/usr/bin/env python3
"""
ToonDB Bulk CLI - Production-grade wrapper for toondb-bulk

Handles:
- Binary resolution with smart fallbacks
- File validation (input exists, output writable)
- Progress reporting
- Error handling with actionable messages
- Platform differences
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List
import tempfile
import json


# ============================================================================
# Constants
# ============================================================================

EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_BINARY_NOT_FOUND = 2
EXIT_INPUT_ERROR = 3
EXIT_OUTPUT_ERROR = 4
EXIT_INTERRUPTED = 130


# ============================================================================
# Binary Resolution
# ============================================================================

def get_bulk_binary() -> str:
    """
    Find the toondb-bulk binary with robust fallback chain.
    
    Resolution order:
    1. TOONDB_BULK_PATH environment variable
    2. Bundled in wheel (_bin/<platform>/)
    3. System PATH
    4. Development build (../target/release/)
    
    Returns:
        Path to the binary
        
    Raises:
        FileNotFoundError: If binary cannot be found
    """
    # 1. Environment variable override
    env_path = os.environ.get("TOONDB_BULK_PATH")
    if env_path:
        if os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path
        raise FileNotFoundError(
            f"TOONDB_BULK_PATH set to '{env_path}' but file not found or not executable"
        )
    
    # Use the existing function from bulk.py if available
    try:
        from toondb.bulk import get_toondb_bulk_path
        return get_toondb_bulk_path()
    except (ImportError, FileNotFoundError):
        pass
    
    # Fallback: manual search
    binary_name = "toondb-bulk.exe" if sys.platform == "win32" else "toondb-bulk"
    
    # Try system PATH
    from shutil import which
    system_binary = which(binary_name)
    if system_binary:
        return system_binary
    
    raise FileNotFoundError(
        f"toondb-bulk binary not found.\n"
        f"Searched:\n"
        f"  - TOONDB_BULK_PATH environment variable\n"
        f"  - Bundled in package (_bin/)\n"
        f"  - System PATH\n"
        f"\nTo fix:\n"
        f"  1. Reinstall: pip install --force-reinstall toondb-client\n"
        f"  2. Or build: cargo build --release -p toondb-bulk\n"
        f"  3. Or set: export TOONDB_BULK_PATH=/path/to/toondb-bulk"
    )


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_input_file(path: str, extensions: Optional[List[str]] = None) -> Path:
    """
    Validate that an input file exists and is readable.
    
    Args:
        path: File path to validate
        extensions: Optional list of valid extensions (e.g., ['.npy', '.raw'])
        
    Returns:
        Resolved Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file isn't readable
        ValueError: If extension is invalid
    """
    file_path = Path(path).resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Input file not readable: {file_path}")
    
    if extensions:
        if file_path.suffix.lower() not in [e.lower() for e in extensions]:
            raise ValueError(
                f"Invalid file extension: {file_path.suffix}\n"
                f"Expected one of: {', '.join(extensions)}"
            )
    
    return file_path


def validate_output_path(path: str, overwrite: bool = False) -> Path:
    """
    Validate that an output path is writable.
    
    Args:
        path: Output file path
        overwrite: Whether to allow overwriting existing files
        
    Returns:
        Resolved Path object
        
    Raises:
        FileExistsError: If file exists and overwrite=False
        PermissionError: If directory isn't writable
    """
    file_path = Path(path).resolve()
    
    # Check if exists
    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {file_path}\n"
            f"Use --overwrite to replace it"
        )
    
    # Ensure parent directory exists and is writable
    parent = file_path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"Cannot create output directory: {parent}")
    
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"Output directory not writable: {parent}")
    
    return file_path


def get_file_size_human(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


# ============================================================================
# Command Handlers
# ============================================================================

def cmd_build_index(binary: str, args: argparse.Namespace) -> int:
    """Handle build-index command with validation."""
    # Validate input
    try:
        input_path = validate_input_file(args.input, extensions=['.npy', '.raw', '.bin'])
        print(f"[Bulk] Input: {input_path} ({get_file_size_human(input_path.stat().st_size)})")
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"[Bulk] Error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    
    # Validate output
    try:
        output_path = validate_output_path(args.output, getattr(args, 'overwrite', False))
        print(f"[Bulk] Output: {output_path}")
    except (FileExistsError, PermissionError) as e:
        print(f"[Bulk] Error: {e}", file=sys.stderr)
        return EXIT_OUTPUT_ERROR
    
    # Build command
    cmd = [
        binary, "build-index",
        "--input", str(input_path),
        "--output", str(output_path),
        "--dimension", str(args.dimension),
    ]
    
    # Optional arguments
    if args.max_connections:
        cmd.extend(["--max-connections", str(args.max_connections)])
    if args.ef_construction:
        cmd.extend(["--ef-construction", str(args.ef_construction)])
    if args.threads is not None:
        cmd.extend(["--threads", str(args.threads)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if getattr(args, 'metric', None):
        cmd.extend(["--metric", args.metric])
    
    # Progress info
    print(f"[Bulk] Building HNSW index (dimension={args.dimension})...")
    
    return run_binary(binary, cmd[1:])  # Skip binary name since run_binary adds it


def cmd_query(binary: str, args: argparse.Namespace) -> int:
    """Handle query command with validation."""
    # Validate index
    try:
        index_path = validate_input_file(args.index, extensions=['.hnsw'])
        print(f"[Bulk] Index: {index_path}")
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"[Bulk] Error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    
    # Validate query vector
    try:
        query_path = validate_input_file(args.query, extensions=['.raw', '.bin', '.npy'])
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"[Bulk] Error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    
    cmd = [
        "query",
        "--index", str(index_path),
        "--query", str(query_path),
        "--k", str(args.k),
    ]
    
    if args.ef:
        cmd.extend(["--ef", str(args.ef)])
    
    return run_binary(binary, cmd)


def cmd_info(binary: str, args: argparse.Namespace) -> int:
    """Handle info command."""
    try:
        index_path = validate_input_file(args.index, extensions=['.hnsw'])
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"[Bulk] Error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    
    return run_binary(binary, ["info", "--index", str(index_path)])


def cmd_convert(binary: str, args: argparse.Namespace) -> int:
    """Handle convert command."""
    try:
        input_path = validate_input_file(args.input)
        print(f"[Bulk] Input: {input_path}")
    except (FileNotFoundError, PermissionError) as e:
        print(f"[Bulk] Error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    
    try:
        output_path = validate_output_path(args.output, getattr(args, 'overwrite', False))
    except (FileExistsError, PermissionError) as e:
        print(f"[Bulk] Error: {e}", file=sys.stderr)
        return EXIT_OUTPUT_ERROR
    
    cmd = [
        "convert",
        "--input", str(input_path),
        "--output", str(output_path),
        "--to-format", args.to_format,
        "--dimension", str(args.dimension),
    ]
    
    return run_binary(binary, cmd)


# ============================================================================
# Binary Execution
# ============================================================================

def run_binary(binary: str, args: List[str]) -> int:
    """
    Run the toondb-bulk binary with the given arguments.
    
    Handles:
    - Keyboard interrupts
    - Process errors
    - stdout/stderr passthrough
    """
    cmd = [binary] + args
    
    try:
        result = subprocess.run(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\n[Bulk] Operation cancelled", file=sys.stderr)
        return EXIT_INTERRUPTED
    except FileNotFoundError:
        print(f"[Bulk] Error: Binary not found: {binary}", file=sys.stderr)
        return EXIT_BINARY_NOT_FOUND
    except PermissionError:
        print(f"[Bulk] Error: Permission denied executing: {binary}", file=sys.stderr)
        return EXIT_GENERAL_ERROR
    except Exception as e:
        print(f"[Bulk] Error: {e}", file=sys.stderr)
        return EXIT_GENERAL_ERROR


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for toondb-bulk CLI."""
    parser = argparse.ArgumentParser(
        prog="toondb-bulk",
        description="ToonDB Bulk Operations - High-performance vector index building and querying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  build-index   Build an HNSW vector index from embeddings
  query         Query an HNSW index for nearest neighbors
  info          Display index metadata
  convert       Convert between vector formats

Examples:
  # Build index from numpy file
  toondb-bulk build-index --input embeddings.npy --output index.hnsw --dimension 768

  # Query index
  toondb-bulk query --index index.hnsw --query query.raw --k 10

  # Get index info
  toondb-bulk info --index index.hnsw

  # Convert numpy to raw format
  toondb-bulk convert --input vectors.npy --output vectors.raw --to-format raw_f32 --dimension 768

Environment Variables:
  TOONDB_BULK_PATH   Path to toondb-bulk binary (overrides bundled)
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # build-index command
    build_parser = subparsers.add_parser("build-index", help="Build HNSW index")
    build_parser.add_argument("--input", "-i", required=True, help="Input vectors file (.npy or .raw)")
    build_parser.add_argument("--output", "-o", required=True, help="Output index path (.hnsw)")
    build_parser.add_argument("--dimension", "-d", type=int, required=True, help="Vector dimension")
    build_parser.add_argument("--max-connections", "-m", type=int, default=16, help="HNSW M parameter (default: 16)")
    build_parser.add_argument("--ef-construction", type=int, default=100, help="HNSW ef_construction (default: 100)")
    build_parser.add_argument("--threads", "-t", type=int, default=0, help="Thread count (0=auto)")
    build_parser.add_argument("--batch-size", type=int, default=1000, help="Batch size (default: 1000)")
    build_parser.add_argument("--metric", choices=["cosine", "l2", "ip"], default="cosine", help="Distance metric")
    build_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    
    # query command
    query_parser = subparsers.add_parser("query", help="Query HNSW index")
    query_parser.add_argument("--index", required=True, help="Index file path (.hnsw)")
    query_parser.add_argument("--query", "-q", required=True, help="Query vector file")
    query_parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors (default: 10)")
    query_parser.add_argument("--ef", type=int, default=64, help="Search ef parameter (default: 64)")
    
    # info command
    info_parser = subparsers.add_parser("info", help="Display index info")
    info_parser.add_argument("--index", required=True, help="Index file path")
    
    # convert command
    convert_parser = subparsers.add_parser("convert", help="Convert vector formats")
    convert_parser.add_argument("--input", "-i", required=True, help="Input file")
    convert_parser.add_argument("--output", "-o", required=True, help="Output file")
    convert_parser.add_argument("--to-format", required=True, choices=["raw_f32", "npy"], help="Target format")
    convert_parser.add_argument("--dimension", "-d", type=int, required=True, help="Vector dimension")
    convert_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    
    # Version
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    
    args = parser.parse_args()
    
    # Version
    if args.version:
        try:
            from toondb import __version__
            print(f"toondb-bulk (Python wrapper) {__version__}")
        except ImportError:
            print("toondb-bulk (Python wrapper)")
        return EXIT_SUCCESS
    
    # Require a command
    if not args.command:
        # Just pass through to native binary for --help etc
        try:
            binary = get_bulk_binary()
            return run_binary(binary, sys.argv[1:])
        except FileNotFoundError as e:
            print(f"[Bulk] Error: {e}", file=sys.stderr)
            return EXIT_BINARY_NOT_FOUND
    
    # Get binary
    try:
        binary = get_bulk_binary()
    except FileNotFoundError as e:
        print(f"[Bulk] Error: {e}", file=sys.stderr)
        return EXIT_BINARY_NOT_FOUND
    
    # Dispatch to command handler
    if args.command == "build-index":
        return cmd_build_index(binary, args)
    elif args.command == "query":
        return cmd_query(binary, args)
    elif args.command == "info":
        return cmd_info(binary, args)
    elif args.command == "convert":
        return cmd_convert(binary, args)
    else:
        # Pass through unknown commands to binary
        return run_binary(binary, sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
