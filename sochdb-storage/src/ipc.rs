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

//! IPC Protocol with Multiplexing and Streaming
//!
//! From mm.md Task 7.1: Pipelined IPC Protocol
//!
//! ## Problem
//!
//! Current IPC is request-response, blocking client during server work:
//! ```text
//! Client: Send Request -> Wait -> Receive Response -> Send Next Request
//! Latency: sum of all request latencies
//! ```
//!
//! ## Solution
//!
//! Pipelining with request IDs and async responses:
//! ```text
//! +------------+     +------------+
//! |   Client   |     |   Server   |
//! +------------+     +------------+
//!       |                   |
//!       |--- Req(id=1) ---->|
//!       |--- Req(id=2) ---->|
//!       |--- Req(id=3) ---->|  (no wait)
//!       |                   |
//!       |<-- Resp(id=2) ----|  (out-of-order OK)
//!       |<-- Resp(id=1) ----|
//!       |<-- Resp(id=3) ----|
//!       |                   |
//!
//! Protocol: Unix domain socket with length-prefixed messages
//! Frame: [4-byte length][request_id: u64][msg_type: u8][payload...]
//! ```
//!
//! ## Benefits
//!
//! - No head-of-line blocking
//! - Batched network I/O
//! - Supports streaming responses for large result sets
//! - Backpressure through flow control

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

/// Request ID for tracking pipelined requests
pub type RequestId = u64;

/// Stream ID for multiplexed streams
pub type StreamId = u64;

/// Message types in the IPC protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    /// Single request expecting single response
    Request = 0,
    /// Single response to a request
    Response = 1,
    /// Request that opens a stream
    StreamStart = 2,
    /// Data chunk in a stream
    StreamData = 3,
    /// End of stream
    StreamEnd = 4,
    /// Error response
    Error = 5,
    /// Flow control: pause sending
    FlowPause = 6,
    /// Flow control: resume sending
    FlowResume = 7,
    /// Ping (keep-alive)
    Ping = 8,
    /// Pong (keep-alive response)
    Pong = 9,
    /// Cancel a pending request/stream
    Cancel = 10,
}

impl TryFrom<u8> for MessageType {
    type Error = IpcError;

    fn try_from(value: u8) -> Result<Self, <Self as TryFrom<u8>>::Error> {
        match value {
            0 => Ok(MessageType::Request),
            1 => Ok(MessageType::Response),
            2 => Ok(MessageType::StreamStart),
            3 => Ok(MessageType::StreamData),
            4 => Ok(MessageType::StreamEnd),
            5 => Ok(MessageType::Error),
            6 => Ok(MessageType::FlowPause),
            7 => Ok(MessageType::FlowResume),
            8 => Ok(MessageType::Ping),
            9 => Ok(MessageType::Pong),
            10 => Ok(MessageType::Cancel),
            _ => Err(IpcError::InvalidMessageType(value)),
        }
    }
}

/// IPC frame header
#[derive(Debug, Clone, Copy)]
pub struct FrameHeader {
    /// Total length of the frame (excluding header)
    pub length: u32,
    /// Request/stream ID
    pub id: u64,
    /// Message type
    pub msg_type: MessageType,
    /// Flags (reserved for future use)
    pub flags: u8,
}

impl FrameHeader {
    /// Header size in bytes
    pub const SIZE: usize = 14; // 4 + 8 + 1 + 1

    /// Maximum payload size (16MB)
    pub const MAX_PAYLOAD: u32 = 16 * 1024 * 1024;

    /// Create a new frame header
    pub fn new(id: u64, msg_type: MessageType, payload_len: usize) -> Self {
        Self {
            length: payload_len as u32,
            id,
            msg_type,
            flags: 0,
        }
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.length.to_le_bytes());
        buf[4..12].copy_from_slice(&self.id.to_le_bytes());
        buf[12] = self.msg_type as u8;
        buf[13] = self.flags;
        buf
    }

    /// Deserialize header from bytes
    pub fn from_bytes(buf: &[u8; Self::SIZE]) -> Result<Self, IpcError> {
        let length = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let id = u64::from_le_bytes([buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11]]);
        let msg_type = MessageType::try_from(buf[12])?;
        let flags = buf[13];

        if length > Self::MAX_PAYLOAD {
            return Err(IpcError::PayloadTooLarge(length as usize));
        }

        Ok(Self {
            length,
            id,
            msg_type,
            flags,
        })
    }
}

/// Complete IPC frame
#[derive(Debug, Clone)]
pub struct Frame {
    pub header: FrameHeader,
    pub payload: Vec<u8>,
}

impl Frame {
    /// Create a request frame
    pub fn request(id: RequestId, payload: Vec<u8>) -> Self {
        Self {
            header: FrameHeader::new(id, MessageType::Request, payload.len()),
            payload,
        }
    }

    /// Create a response frame
    pub fn response(id: RequestId, payload: Vec<u8>) -> Self {
        Self {
            header: FrameHeader::new(id, MessageType::Response, payload.len()),
            payload,
        }
    }

    /// Create a stream start frame
    pub fn stream_start(id: StreamId, payload: Vec<u8>) -> Self {
        Self {
            header: FrameHeader::new(id, MessageType::StreamStart, payload.len()),
            payload,
        }
    }

    /// Create a stream data frame
    pub fn stream_data(id: StreamId, payload: Vec<u8>) -> Self {
        Self {
            header: FrameHeader::new(id, MessageType::StreamData, payload.len()),
            payload,
        }
    }

    /// Create a stream end frame
    pub fn stream_end(id: StreamId) -> Self {
        Self {
            header: FrameHeader::new(id, MessageType::StreamEnd, 0),
            payload: Vec::new(),
        }
    }

    /// Create an error frame
    pub fn error(id: RequestId, error_code: u32, message: &str) -> Self {
        let mut payload = Vec::with_capacity(4 + message.len());
        payload.extend_from_slice(&error_code.to_le_bytes());
        payload.extend_from_slice(message.as_bytes());
        Self {
            header: FrameHeader::new(id, MessageType::Error, payload.len()),
            payload,
        }
    }

    /// Create a ping frame
    pub fn ping(id: RequestId) -> Self {
        Self {
            header: FrameHeader::new(id, MessageType::Ping, 0),
            payload: Vec::new(),
        }
    }

    /// Create a pong frame
    pub fn pong(id: RequestId) -> Self {
        Self {
            header: FrameHeader::new(id, MessageType::Pong, 0),
            payload: Vec::new(),
        }
    }

    /// Create a cancel frame
    pub fn cancel(id: RequestId) -> Self {
        Self {
            header: FrameHeader::new(id, MessageType::Cancel, 0),
            payload: Vec::new(),
        }
    }

    /// Serialize frame to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(FrameHeader::SIZE + self.payload.len());
        buf.extend_from_slice(&self.header.to_bytes());
        buf.extend_from_slice(&self.payload);
        buf
    }
}

/// IPC error types
#[derive(Debug)]
pub enum IpcError {
    Io(io::Error),
    InvalidMessageType(u8),
    PayloadTooLarge(usize),
    UnexpectedEof,
    RequestCancelled(RequestId),
    StreamClosed(StreamId),
    Timeout,
}

impl From<io::Error> for IpcError {
    fn from(e: io::Error) -> Self {
        IpcError::Io(e)
    }
}

impl std::fmt::Display for IpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IpcError::Io(e) => write!(f, "IO error: {}", e),
            IpcError::InvalidMessageType(t) => write!(f, "Invalid message type: {}", t),
            IpcError::PayloadTooLarge(size) => write!(f, "Payload too large: {} bytes", size),
            IpcError::UnexpectedEof => write!(f, "Unexpected end of stream"),
            IpcError::RequestCancelled(id) => write!(f, "Request {} cancelled", id),
            IpcError::StreamClosed(id) => write!(f, "Stream {} closed", id),
            IpcError::Timeout => write!(f, "Operation timed out"),
        }
    }
}

impl std::error::Error for IpcError {}

/// Frame reader for parsing incoming frames
pub struct FrameReader<R: Read> {
    reader: R,
    header_buf: [u8; FrameHeader::SIZE],
}

impl<R: Read> FrameReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            header_buf: [0u8; FrameHeader::SIZE],
        }
    }

    /// Read the next frame
    pub fn read_frame(&mut self) -> Result<Frame, IpcError> {
        // Read header
        self.reader.read_exact(&mut self.header_buf)?;
        let header = FrameHeader::from_bytes(&self.header_buf)?;

        // Read payload
        let mut payload = vec![0u8; header.length as usize];
        self.reader.read_exact(&mut payload)?;

        Ok(Frame { header, payload })
    }

    /// Get inner reader
    pub fn into_inner(self) -> R {
        self.reader
    }
}

/// Frame writer for serializing outgoing frames
pub struct FrameWriter<W: Write> {
    writer: W,
    /// Write buffer for batching
    buffer: Vec<u8>,
    /// Maximum buffer size before flush
    max_buffer: usize,
}

impl<W: Write> FrameWriter<W> {
    /// Default buffer size (64KB)
    const DEFAULT_BUFFER: usize = 64 * 1024;

    pub fn new(writer: W) -> Self {
        Self {
            writer,
            buffer: Vec::with_capacity(Self::DEFAULT_BUFFER),
            max_buffer: Self::DEFAULT_BUFFER,
        }
    }

    /// Write a frame (may buffer)
    pub fn write_frame(&mut self, frame: &Frame) -> Result<(), IpcError> {
        let bytes = frame.to_bytes();

        // If frame is larger than buffer, flush and write directly
        if bytes.len() > self.max_buffer {
            self.flush()?;
            self.writer.write_all(&bytes)?;
            return Ok(());
        }

        // If buffer would overflow, flush first
        if self.buffer.len() + bytes.len() > self.max_buffer {
            self.flush()?;
        }

        self.buffer.extend_from_slice(&bytes);
        Ok(())
    }

    /// Flush buffered frames
    pub fn flush(&mut self) -> Result<(), IpcError> {
        if !self.buffer.is_empty() {
            self.writer.write_all(&self.buffer)?;
            self.buffer.clear();
        }
        self.writer.flush()?;
        Ok(())
    }

    /// Get inner writer
    pub fn into_inner(self) -> W {
        self.writer
    }
}

/// Pending request tracker
struct PendingRequest {
    callback: Box<dyn FnOnce(Result<Frame, IpcError>) + Send>,
}

/// Request multiplexer for pipelining
pub struct RequestMultiplexer {
    /// Next request ID
    next_id: AtomicU64,
    /// Pending requests waiting for responses
    pending: Mutex<HashMap<RequestId, PendingRequest>>,
    /// Active streams
    streams: Mutex<HashMap<StreamId, StreamState>>,
}

/// State of an active stream
struct StreamState {
    /// Callback for each data chunk
    on_data: Box<dyn Fn(Vec<u8>) + Send>,
    /// Callback when stream ends
    on_end: Box<dyn FnOnce() + Send>,
    /// Whether flow control is paused
    #[allow(dead_code)]
    paused: bool,
}

impl Default for RequestMultiplexer {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestMultiplexer {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
            streams: Mutex::new(HashMap::new()),
        }
    }

    /// Allocate a new request ID
    pub fn next_id(&self) -> RequestId {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Register a pending request
    pub fn register_request<F>(&self, id: RequestId, callback: F)
    where
        F: FnOnce(Result<Frame, IpcError>) + Send + 'static,
    {
        self.pending.lock().insert(
            id,
            PendingRequest {
                callback: Box::new(callback),
            },
        );
    }

    /// Register a stream
    pub fn register_stream<D, E>(&self, id: StreamId, on_data: D, on_end: E)
    where
        D: Fn(Vec<u8>) + Send + 'static,
        E: FnOnce() + Send + 'static,
    {
        self.streams.lock().insert(
            id,
            StreamState {
                on_data: Box::new(on_data),
                on_end: Box::new(on_end),
                paused: false,
            },
        );
    }

    /// Handle an incoming frame
    pub fn handle_frame(&self, frame: Frame) {
        match frame.header.msg_type {
            MessageType::Response | MessageType::Error => {
                if let Some(pending) = self.pending.lock().remove(&frame.header.id) {
                    (pending.callback)(Ok(frame));
                }
            }
            MessageType::StreamData => {
                if let Some(state) = self.streams.lock().get(&frame.header.id) {
                    (state.on_data)(frame.payload);
                }
            }
            MessageType::StreamEnd => {
                if let Some(state) = self.streams.lock().remove(&frame.header.id) {
                    (state.on_end)();
                }
            }
            MessageType::Pong => {
                // Ping/pong handled separately
            }
            _ => {
                // Request types are handled by server
            }
        }
    }

    /// Cancel a pending request
    pub fn cancel(&self, id: RequestId) {
        if let Some(pending) = self.pending.lock().remove(&id) {
            (pending.callback)(Err(IpcError::RequestCancelled(id)));
        }
        if let Some(state) = self.streams.lock().remove(&id) {
            (state.on_end)();
        }
    }

    /// Get number of pending requests
    pub fn pending_count(&self) -> usize {
        self.pending.lock().len()
    }
}

/// Batch request builder for efficient pipelining
pub struct BatchRequest {
    requests: Vec<(RequestId, Vec<u8>)>,
}

impl Default for BatchRequest {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchRequest {
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
        }
    }

    /// Add a request to the batch
    pub fn add(&mut self, id: RequestId, payload: Vec<u8>) -> &mut Self {
        self.requests.push((id, payload));
        self
    }

    /// Build frames for all requests
    pub fn build(self) -> Vec<Frame> {
        self.requests
            .into_iter()
            .map(|(id, payload)| Frame::request(id, payload))
            .collect()
    }

    /// Get the number of requests
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

/// Flow control state
#[derive(Debug, Clone)]
pub struct FlowControl {
    /// Window size (max outstanding bytes)
    pub window_size: usize,
    /// Current outstanding bytes
    pub outstanding: usize,
    /// Whether paused
    pub paused: bool,
}

impl Default for FlowControl {
    fn default() -> Self {
        Self {
            window_size: 64 * 1024, // 64KB default window
            outstanding: 0,
            paused: false,
        }
    }
}

impl FlowControl {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            outstanding: 0,
            paused: false,
        }
    }

    /// Check if we can send more data
    pub fn can_send(&self) -> bool {
        !self.paused && self.outstanding < self.window_size
    }

    /// Record sent bytes
    pub fn record_sent(&mut self, bytes: usize) {
        self.outstanding += bytes;
        if self.outstanding >= self.window_size {
            self.paused = true;
        }
    }

    /// Record acknowledged bytes
    pub fn record_acked(&mut self, bytes: usize) {
        self.outstanding = self.outstanding.saturating_sub(bytes);
        if self.outstanding < self.window_size / 2 {
            self.paused = false;
        }
    }

    /// Pause sending
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resume sending
    pub fn resume(&mut self) {
        self.paused = false;
    }
}

/// Stream response writer for sending chunked results
pub struct StreamWriter<W: Write> {
    writer: Arc<Mutex<FrameWriter<W>>>,
    stream_id: StreamId,
    flow_control: FlowControl,
}

impl<W: Write> StreamWriter<W> {
    pub fn new(writer: Arc<Mutex<FrameWriter<W>>>, stream_id: StreamId) -> Self {
        Self {
            writer,
            stream_id,
            flow_control: FlowControl::default(),
        }
    }

    /// Write a chunk of data
    pub fn write_chunk(&mut self, data: Vec<u8>) -> Result<(), IpcError> {
        // Wait for flow control if needed
        while !self.flow_control.can_send() {
            std::thread::yield_now();
        }

        let frame = Frame::stream_data(self.stream_id, data);
        let size = frame.payload.len();

        self.writer.lock().write_frame(&frame)?;
        self.flow_control.record_sent(size);

        Ok(())
    }

    /// End the stream
    pub fn finish(self) -> Result<(), IpcError> {
        let frame = Frame::stream_end(self.stream_id);
        let mut writer = self.writer.lock();
        writer.write_frame(&frame)?;
        writer.flush()
    }
}

/// Request handler trait for server-side processing
pub trait RequestHandler: Send + Sync {
    /// Handle a single request
    fn handle_request(&self, request_id: RequestId, payload: &[u8]) -> Result<Vec<u8>, IpcError>;

    /// Handle a streaming request
    fn handle_stream<W: Write>(
        &self,
        stream_id: StreamId,
        payload: &[u8],
        writer: StreamWriter<W>,
    ) -> Result<(), IpcError>;
}

/// IPC server for handling incoming connections
pub struct IpcServer<H: RequestHandler> {
    handler: Arc<H>,
}

impl<H: RequestHandler> IpcServer<H> {
    pub fn new(handler: H) -> Self {
        Self {
            handler: Arc::new(handler),
        }
    }

    /// Process frames from a connection
    pub fn process<R: Read, W: Write>(
        &self,
        reader: &mut FrameReader<R>,
        writer: Arc<Mutex<FrameWriter<W>>>,
    ) -> Result<(), IpcError> {
        loop {
            let frame = match reader.read_frame() {
                Ok(f) => f,
                Err(IpcError::Io(e)) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    return Ok(()); // Connection closed
                }
                Err(e) => return Err(e),
            };

            match frame.header.msg_type {
                MessageType::Request => {
                    let response = match self.handler.handle_request(frame.header.id, &frame.payload)
                    {
                        Ok(data) => Frame::response(frame.header.id, data),
                        Err(e) => Frame::error(frame.header.id, 1, &e.to_string()),
                    };
                    writer.lock().write_frame(&response)?;
                }
                MessageType::StreamStart => {
                    let stream_writer = StreamWriter::new(Arc::clone(&writer), frame.header.id);
                    if let Err(e) =
                        self.handler
                            .handle_stream(frame.header.id, &frame.payload, stream_writer)
                    {
                        let err = Frame::error(frame.header.id, 2, &e.to_string());
                        writer.lock().write_frame(&err)?;
                    }
                }
                MessageType::Ping => {
                    let pong = Frame::pong(frame.header.id);
                    writer.lock().write_frame(&pong)?;
                }
                MessageType::Cancel => {
                    // Cancel handling would be implemented with cancellation tokens
                }
                _ => {
                    // Ignore client-side message types
                }
            }

            // Flush periodically
            writer.lock().flush()?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_frame_header_roundtrip() {
        let header = FrameHeader::new(12345, MessageType::Request, 100);
        let bytes = header.to_bytes();
        let parsed = FrameHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.id, 12345);
        assert_eq!(parsed.msg_type, MessageType::Request);
        assert_eq!(parsed.length, 100);
    }

    #[test]
    fn test_frame_roundtrip() {
        let original = Frame::request(1, b"hello world".to_vec());
        let bytes = original.to_bytes();

        let mut reader = FrameReader::new(Cursor::new(bytes));
        let parsed = reader.read_frame().unwrap();

        assert_eq!(parsed.header.id, 1);
        assert_eq!(parsed.header.msg_type, MessageType::Request);
        assert_eq!(parsed.payload, b"hello world");
    }

    #[test]
    fn test_batch_request() {
        let mut batch = BatchRequest::new();
        batch.add(1, b"request1".to_vec());
        batch.add(2, b"request2".to_vec());
        batch.add(3, b"request3".to_vec());

        let frames = batch.build();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].header.id, 1);
        assert_eq!(frames[1].header.id, 2);
        assert_eq!(frames[2].header.id, 3);
    }

    #[test]
    fn test_multiplexer() {
        let mux = RequestMultiplexer::new();

        let id1 = mux.next_id();
        let id2 = mux.next_id();

        assert_ne!(id1, id2);

        use std::sync::atomic::AtomicBool;

        let received1 = Arc::new(AtomicBool::new(false));
        let received2 = Arc::new(AtomicBool::new(false));

        {
            let r1 = Arc::clone(&received1);
            mux.register_request(id1, move |_| {
                r1.store(true, Ordering::SeqCst);
            });
        }

        {
            let r2 = Arc::clone(&received2);
            mux.register_request(id2, move |_| {
                r2.store(true, Ordering::SeqCst);
            });
        }

        // Handle response for id2 first (out of order)
        mux.handle_frame(Frame::response(id2, b"resp2".to_vec()));
        assert!(!received1.load(Ordering::SeqCst));
        assert!(received2.load(Ordering::SeqCst));

        // Handle response for id1
        mux.handle_frame(Frame::response(id1, b"resp1".to_vec()));
        assert!(received1.load(Ordering::SeqCst));
    }

    #[test]
    fn test_flow_control() {
        let mut fc = FlowControl::new(100);

        assert!(fc.can_send());

        fc.record_sent(50);
        assert!(fc.can_send());
        assert_eq!(fc.outstanding, 50);

        fc.record_sent(60);
        assert!(!fc.can_send()); // Exceeded window
        assert!(fc.paused);

        fc.record_acked(80);
        assert!(fc.can_send()); // Below half window
        assert!(!fc.paused);
    }

    #[test]
    fn test_error_frame() {
        let frame = Frame::error(42, 500, "Internal error");

        assert_eq!(frame.header.id, 42);
        assert_eq!(frame.header.msg_type, MessageType::Error);

        // Parse error code and message
        let error_code = u32::from_le_bytes([
            frame.payload[0],
            frame.payload[1],
            frame.payload[2],
            frame.payload[3],
        ]);
        let message = std::str::from_utf8(&frame.payload[4..]).unwrap();

        assert_eq!(error_code, 500);
        assert_eq!(message, "Internal error");
    }
}
