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

//! JSON-RPC message framing for MCP stdio transport
//!
//! MCP uses LSP-style framing:
//! ```text
//! Content-Length: <N>\r\n
//! \r\n
//! <N bytes of JSON>
//! ```

use std::io::{self, BufRead, Write};

/// Read a framed message from the reader.
/// Supports both Content-Length framing (LSP-style) and NDJSON (newline-delimited JSON).
/// Returns Ok(None) on EOF, or Ok(Some((bytes, format))) on success.
pub fn read_message<R: BufRead>(reader: &mut R) -> io::Result<Option<(Vec<u8>, WireFormat)>> {
    let mut first_line = String::new();
    let bytes_read = reader.read_line(&mut first_line)?;

    if bytes_read == 0 {
        return Ok(None); // EOF
    }

    let trimmed = first_line.trim();

    // Detect format: Content-Length header vs raw JSON
    if trimmed.starts_with("Content-Length:") {
        // LSP-style framing
        let content_length: usize = trimmed
            .strip_prefix("Content-Length:")
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Invalid Content-Length header")
            })?;

        // Read remaining headers until empty line
        loop {
            let mut header_line = String::new();
            let header_bytes = reader.read_line(&mut header_line)?;
            if header_bytes == 0 {
                return Ok(None); // Unexpected EOF
            }
            if header_line.trim().is_empty() {
                break; // End of headers
            }
            // Ignore other headers (Content-Type, etc.)
        }

        // Read body
        let mut body = vec![0u8; content_length];
        reader.read_exact(&mut body)?;

        Ok(Some((body, WireFormat::ContentLength)))
    } else if trimmed.starts_with('{') {
        // NDJSON format: raw JSON on a line
        Ok(Some((trimmed.as_bytes().to_vec(), WireFormat::Ndjson)))
    } else if trimmed.is_empty() {
        // Empty line, try reading next line
        read_message(reader)
    } else {
        // Unknown format, treat as error
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown message format, expected Content-Length or JSON, got: {}", 
                    if trimmed.len() > 50 { &trimmed[..50] } else { trimmed })
        ))
    }
}

/// Wire format for MCP messages
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WireFormat {
    /// LSP-style Content-Length framing
    ContentLength,
    /// Newline-delimited JSON (one JSON object per line)
    Ndjson,
}

/// Write a framed message to the writer using the specified format.
pub fn write_message_format<W: Write, T: serde::Serialize>(
    writer: &mut W,
    message: &T,
    format: WireFormat,
) -> io::Result<()> {
    let body = serde_json::to_vec(message)?;

    match format {
        WireFormat::ContentLength => {
            write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
            writer.write_all(&body)?;
        }
        WireFormat::Ndjson => {
            writer.write_all(&body)?;
            writer.write_all(b"\n")?;
        }
    }

    Ok(())
}

/// Write a framed message to the writer.
#[allow(dead_code)]
pub fn write_message<W: Write, T: serde::Serialize>(writer: &mut W, message: &T) -> io::Result<()> {
    let body = serde_json::to_vec(message)?;

    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_message() {
        let input = b"Content-Length: 13\r\n\r\n{\"test\": 123}";
        let mut reader = Cursor::new(input);

        let (msg, format) = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(msg, b"{\"test\": 123}");
        assert_eq!(format, WireFormat::ContentLength);
    }

    #[test]
    fn test_write_message() {
        let mut output = Vec::new();
        let msg = serde_json::json!({"test": 123});

        write_message(&mut output, &msg).unwrap();

        let expected = b"Content-Length: 12\r\n\r\n{\"test\":123}";
        assert_eq!(output, expected);
    }
}
