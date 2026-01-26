// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Key types for SochDB indexing

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::cmp::Ordering;
use std::io::Result as IoResult;

/// Composite key for temporal ordering
///
/// Primary: timestamp_us (microseconds for temporal queries)
/// Secondary: record_id (for uniqueness within same timestamp)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemporalKey {
    pub timestamp_us: u64,
    pub edge_id: u128,
}

impl TemporalKey {
    pub fn new(timestamp_us: u64, edge_id: u128) -> Self {
        Self {
            timestamp_us,
            edge_id,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24);
        buf.write_u64::<LittleEndian>(self.timestamp_us).unwrap();
        buf.write_u128::<LittleEndian>(self.edge_id).unwrap();
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> IoResult<Self> {
        let mut cursor = bytes;
        let timestamp_us = cursor.read_u64::<LittleEndian>()?;
        let edge_id = cursor.read_u128::<LittleEndian>()?;
        Ok(Self {
            timestamp_us,
            edge_id,
        })
    }
}

impl PartialOrd for TemporalKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TemporalKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp_us
            .cmp(&other.timestamp_us)
            .then_with(|| self.edge_id.cmp(&other.edge_id))
    }
}

/// Causal key for graph traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CausalKey {
    pub parent_id: u128,
    pub child_id: u128,
}

impl CausalKey {
    pub fn new(parent_id: u128, child_id: u128) -> Self {
        Self {
            parent_id,
            child_id,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.write_u128::<LittleEndian>(self.parent_id).unwrap();
        buf.write_u128::<LittleEndian>(self.child_id).unwrap();
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> IoResult<Self> {
        let mut cursor = bytes;
        let parent_id = cursor.read_u128::<LittleEndian>()?;
        let child_id = cursor.read_u128::<LittleEndian>()?;
        Ok(Self {
            parent_id,
            child_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_key_ordering() {
        let k1 = TemporalKey::new(100, 1);
        let k2 = TemporalKey::new(100, 2);
        let k3 = TemporalKey::new(200, 1);

        assert!(k1 < k2);
        assert!(k2 < k3);
        assert!(k1 < k3);
    }

    #[test]
    fn test_temporal_key_serialization() {
        let key = TemporalKey::new(12345, 67890);
        let bytes = key.to_bytes();
        let decoded = TemporalKey::from_bytes(&bytes).unwrap();
        assert_eq!(key, decoded);
    }
}
