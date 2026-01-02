# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ToonDB Python SDK

A Python client for ToonDB - the database optimized for LLM context retrieval.

Provides two modes of access:
- Embedded: Direct database access via FFI (single process)
- IPC: Client-server access via Unix sockets (multi-process)
- Vector: HNSW vector search (15x faster than ChromaDB)
"""

from .ipc_client import IpcClient
from .database import Database, Transaction
from .query import Query, SQLQueryResult
from .errors import ToonDBError, ConnectionError, TransactionError, ProtocolError

# Vector search (optional - requires libtoondb_index)
try:
    from .vector import VectorIndex
except ImportError:
    VectorIndex = None

# Bulk operations (optional - requires toondb-bulk binary)
try:
    from .bulk import bulk_build_index, bulk_query_index, BulkBuildStats, QueryResult
except ImportError:
    bulk_build_index = None
    bulk_query_index = None
    BulkBuildStats = None
    QueryResult = None

__version__ = "0.2.7"
__all__ = [
    "Database",
    "Transaction", 
    "Query",
    "SQLQueryResult",
    "IpcClient",
    "VectorIndex",
    "bulk_build_index",
    "bulk_query_index",
    "BulkBuildStats",
    "QueryResult",
    "ToonDBError",
    "ConnectionError",
    "TransactionError",
    "ProtocolError",
]
