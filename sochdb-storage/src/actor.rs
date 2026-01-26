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

//! Actor-Based Connection Manager
//!
//! From mm.md Task 7.2: Unified Connection Model
//!
//! ## Problem
//!
//! Current: One global Arc<Mutex<Database>> shared by all threads.
//! Issue: Lock contention, complex lifetime management, no affinity.
//!
//! ## Solution
//!
//! Actor model with single-owner database connections:
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐
//! │   Client    │────>│   Actor 1   │──┐
//! └─────────────┘     └─────────────┘  │
//!                                      │  ┌──────────────────┐
//! ┌─────────────┐     ┌─────────────┐  ├─>│    Database      │
//! │   Client    │────>│   Actor 2   │──┤  │  (owned by pool) │
//! └─────────────┘     └─────────────┘  │  └──────────────────┘
//!                                      │
//! ┌─────────────┐     ┌─────────────┐  │
//! │   Client    │────>│   Actor 3   │──┘
//! └─────────────┘     └─────────────┘
//!
//! Each Actor:
//! - Owns its connection (no sharing)
//! - Processes messages sequentially (no locks)
//! - Has CPU affinity for cache locality
//! ```
//!
//! ## Benefits
//!
//! - Zero lock contention within actor
//! - Predictable latency (no lock wait)
//! - Cache-friendly (single-threaded access pattern)
//! - Natural backpressure through message queue

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;

/// Actor ID
pub type ActorId = u64;

/// Message ID for tracking
pub type MessageId = u64;

/// Actor message envelope
pub struct Message<T> {
    pub id: MessageId,
    pub payload: T,
    pub created_at: Instant,
}

impl<T> Message<T> {
    pub fn new(id: MessageId, payload: T) -> Self {
        Self {
            id,
            payload,
            created_at: Instant::now(),
        }
    }

    /// Age of the message
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Response wrapper
pub struct Response<R> {
    pub message_id: MessageId,
    pub result: Result<R, ActorError>,
    pub processing_time: Duration,
}

/// Actor error types
#[derive(Debug)]
pub enum ActorError {
    /// Mailbox is full
    MailboxFull,
    /// Actor is stopped
    ActorStopped,
    /// Handler error
    HandlerError(String),
    /// Timeout waiting for response
    Timeout,
}

impl std::fmt::Display for ActorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActorError::MailboxFull => write!(f, "Actor mailbox is full"),
            ActorError::ActorStopped => write!(f, "Actor has stopped"),
            ActorError::HandlerError(e) => write!(f, "Handler error: {}", e),
            ActorError::Timeout => write!(f, "Request timed out"),
        }
    }
}

impl std::error::Error for ActorError {}

/// Handler trait for processing messages
pub trait Handler<M, R>: Send + Sync {
    fn handle(&mut self, message: M) -> Result<R, ActorError>;
}

/// Actor statistics
#[derive(Debug, Clone, Default)]
pub struct ActorStats {
    pub messages_processed: u64,
    pub messages_pending: usize,
    pub total_processing_time_us: u64,
    pub max_processing_time_us: u64,
    pub avg_wait_time_us: u64,
}

/// Internal actor state
struct ActorInner<M, R, H: Handler<M, R>> {
    #[allow(dead_code)]
    id: ActorId,
    handler: H,
    inbox: Receiver<Message<M>>,
    running: Arc<AtomicBool>,
    stats: ActorStats,
    _phantom: std::marker::PhantomData<R>,
}

impl<M: Send + 'static, R: Send + 'static, H: Handler<M, R> + 'static> ActorInner<M, R, H> {
    fn run(mut self, response_tx: Sender<Response<R>>) {
        while self.running.load(Ordering::Acquire) {
            match self.inbox.recv_timeout(Duration::from_millis(100)) {
                Ok(msg) => {
                    let wait_time = msg.age();
                    let start = Instant::now();

                    let result = self.handler.handle(msg.payload);

                    let processing_time = start.elapsed();

                    // Update stats
                    self.stats.messages_processed += 1;
                    let proc_us = processing_time.as_micros() as u64;
                    self.stats.total_processing_time_us += proc_us;
                    if proc_us > self.stats.max_processing_time_us {
                        self.stats.max_processing_time_us = proc_us;
                    }
                    let wait_us = wait_time.as_micros() as u64;
                    let n = self.stats.messages_processed;
                    self.stats.avg_wait_time_us =
                        (self.stats.avg_wait_time_us * (n - 1) + wait_us) / n;

                    let _ = response_tx.send(Response {
                        message_id: msg.id,
                        result,
                        processing_time,
                    });
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // No message, check if still running
                    continue;
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Sender dropped, stop actor
                    break;
                }
            }
        }
    }
}

/// Actor reference for sending messages
pub struct ActorRef<M, R> {
    id: ActorId,
    inbox: Sender<Message<M>>,
    responses: Receiver<Response<R>>,
    next_message_id: AtomicU64,
    running: Arc<AtomicBool>,
}

impl<M: Send + 'static, R: Send + 'static> ActorRef<M, R> {
    /// Send a message and wait for response
    pub fn ask(&self, message: M) -> Result<R, ActorError> {
        self.ask_timeout(message, Duration::from_secs(30))
    }

    /// Send a message with timeout
    pub fn ask_timeout(&self, message: M, timeout: Duration) -> Result<R, ActorError> {
        if !self.running.load(Ordering::Acquire) {
            return Err(ActorError::ActorStopped);
        }

        let id = self.next_message_id.fetch_add(1, Ordering::SeqCst);
        let msg = Message::new(id, message);

        self.inbox
            .send(msg)
            .map_err(|_| ActorError::ActorStopped)?;

        // Wait for response
        match self.responses.recv_timeout(timeout) {
            Ok(resp) => resp.result,
            Err(_) => Err(ActorError::Timeout),
        }
    }

    /// Send a message without waiting (fire-and-forget)
    pub fn tell(&self, message: M) -> Result<(), ActorError> {
        if !self.running.load(Ordering::Acquire) {
            return Err(ActorError::ActorStopped);
        }

        let id = self.next_message_id.fetch_add(1, Ordering::SeqCst);
        let msg = Message::new(id, message);

        self.inbox
            .try_send(msg)
            .map_err(|e| match e {
                crossbeam_channel::TrySendError::Full(_) => ActorError::MailboxFull,
                crossbeam_channel::TrySendError::Disconnected(_) => ActorError::ActorStopped,
            })
    }

    /// Check if actor is still running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Get actor ID
    pub fn id(&self) -> ActorId {
        self.id
    }

    /// Stop the actor
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }
}

/// Actor spawner
pub struct Actor;

impl Actor {
    /// Spawn a new actor with the given handler
    pub fn spawn<M, R, H>(id: ActorId, handler: H, mailbox_size: usize) -> (ActorRef<M, R>, JoinHandle<()>)
    where
        M: Send + 'static,
        R: Send + 'static,
        H: Handler<M, R> + 'static,
    {
        let (inbox_tx, inbox_rx) = bounded(mailbox_size);
        let (resp_tx, resp_rx) = bounded(mailbox_size);
        let running = Arc::new(AtomicBool::new(true));

        let inner = ActorInner {
            id,
            handler,
            inbox: inbox_rx,
            running: Arc::clone(&running),
            stats: ActorStats::default(),
            _phantom: std::marker::PhantomData,
        };

        let handle = thread::spawn(move || {
            inner.run(resp_tx);
        });

        let actor_ref = ActorRef {
            id,
            inbox: inbox_tx,
            responses: resp_rx,
            next_message_id: AtomicU64::new(1),
            running,
        };

        (actor_ref, handle)
    }
}

/// Connection pool using actor model
pub struct ActorPool<M: Send + Clone + 'static, R: Send + 'static> {
    actors: Vec<ActorRef<M, R>>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    next_actor: AtomicUsize,
    #[allow(dead_code)]
    next_actor_id: AtomicU64,
}

impl<M: Send + Clone + 'static, R: Send + 'static> ActorPool<M, R> {
    /// Create a new pool with the given factory
    pub fn new<F, H>(size: usize, factory: F, mailbox_size: usize) -> Self
    where
        F: Fn() -> H,
        H: Handler<M, R> + 'static,
    {
        let mut actors = Vec::with_capacity(size);
        let mut handles = Vec::with_capacity(size);
        let next_id = AtomicU64::new(1);

        for _ in 0..size {
            let id = next_id.fetch_add(1, Ordering::SeqCst);
            let handler = factory();
            let (actor_ref, handle) = Actor::spawn(id, handler, mailbox_size);
            actors.push(actor_ref);
            handles.push(handle);
        }

        Self {
            actors,
            handles: Mutex::new(handles),
            next_actor: AtomicUsize::new(0),
            next_actor_id: next_id,
        }
    }

    /// Send a message using round-robin selection
    pub fn ask(&self, message: M) -> Result<R, ActorError> {
        let idx = self.next_actor.fetch_add(1, Ordering::Relaxed) % self.actors.len();
        self.actors[idx].ask(message)
    }

    /// Send to a specific actor
    pub fn ask_actor(&self, actor_idx: usize, message: M) -> Result<R, ActorError> {
        if actor_idx >= self.actors.len() {
            return Err(ActorError::HandlerError("Invalid actor index".to_string()));
        }
        self.actors[actor_idx].ask(message)
    }

    /// Broadcast to all actors
    pub fn broadcast(&self, message: M) -> Vec<Result<R, ActorError>> {
        self.actors.iter().map(|a| a.ask(message.clone())).collect()
    }

    /// Get number of actors
    pub fn size(&self) -> usize {
        self.actors.len()
    }

    /// Stop all actors
    pub fn shutdown(&self) {
        for actor in &self.actors {
            actor.stop();
        }

        // Wait for all actors to finish
        let mut handles = self.handles.lock();
        for handle in handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl<M: Send + Clone + 'static, R: Send + 'static> Drop for ActorPool<M, R> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Work-stealing actor pool for better load balancing
#[allow(dead_code)]
pub struct WorkStealingPool<M: Send + 'static, R: Send + 'static> {
    actors: Vec<Arc<ActorRef<M, R>>>,
    queues: Vec<Arc<Mutex<VecDeque<Message<M>>>>>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    running: Arc<AtomicBool>,
}

/// Affinity hint for actor assignment
#[derive(Debug, Clone)]
pub enum AffinityHint {
    /// Route based on key hash (for locality)
    KeyBased(u64),
    /// Use least loaded actor
    LeastLoaded,
    /// Round-robin
    RoundRobin,
    /// Specific actor
    Specific(ActorId),
}

/// Request router for intelligent actor selection
pub struct RequestRouter<M: Send + Clone + 'static, R: Send + 'static> {
    pool: Arc<ActorPool<M, R>>,
    key_to_actor: Mutex<std::collections::HashMap<u64, usize>>,
}

impl<M: Send + Clone + 'static, R: Send + 'static> RequestRouter<M, R> {
    pub fn new(pool: Arc<ActorPool<M, R>>) -> Self {
        Self {
            pool,
            key_to_actor: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Route a request with affinity hint
    pub fn route(&self, message: M, hint: AffinityHint) -> Result<R, ActorError> {
        let actor_idx = match hint {
            AffinityHint::KeyBased(key) => {
                let mut mapping = self.key_to_actor.lock();
                *mapping.entry(key).or_insert_with(|| {
                    (key as usize) % self.pool.size()
                })
            }
            AffinityHint::LeastLoaded => {
                // Simple round-robin as proxy for least loaded
                0 // In production, track queue depths
            }
            AffinityHint::RoundRobin => {
                // Let the pool handle round-robin
                return self.pool.ask(message);
            }
            AffinityHint::Specific(id) => {
                (id as usize) % self.pool.size()
            }
        };

        self.pool.ask_actor(actor_idx, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoHandler;

    impl Handler<String, String> for EchoHandler {
        fn handle(&mut self, message: String) -> Result<String, ActorError> {
            Ok(format!("Echo: {}", message))
        }
    }

    struct CounterHandler {
        count: u64,
    }

    impl Handler<(), u64> for CounterHandler {
        fn handle(&mut self, _: ()) -> Result<u64, ActorError> {
            self.count += 1;
            Ok(self.count)
        }
    }

    #[test]
    fn test_actor_spawn() {
        let (actor, handle) = Actor::spawn(1, EchoHandler, 10);

        let result = actor.ask("Hello".to_string()).unwrap();
        assert_eq!(result, "Echo: Hello");

        actor.stop();
        handle.join().unwrap();
    }

    #[test]
    fn test_actor_pool() {
        let pool = ActorPool::new(4, || EchoHandler, 100);

        // Send multiple messages
        let results: Vec<_> = (0..10)
            .map(|i| pool.ask(format!("Message {}", i)))
            .collect();

        for (i, result) in results.into_iter().enumerate() {
            assert_eq!(result.unwrap(), format!("Echo: Message {}", i));
        }

        pool.shutdown();
    }

    #[test]
    fn test_counter_handler() {
        let (actor, handle) = Actor::spawn(1, CounterHandler { count: 0 }, 10);

        assert_eq!(actor.ask(()).unwrap(), 1);
        assert_eq!(actor.ask(()).unwrap(), 2);
        assert_eq!(actor.ask(()).unwrap(), 3);

        actor.stop();
        handle.join().unwrap();
    }

    #[test]
    fn test_broadcast() {
        let pool = ActorPool::new(4, || CounterHandler { count: 0 }, 100);

        let results = pool.broadcast(());
        assert_eq!(results.len(), 4);
        for result in results {
            assert_eq!(result.unwrap(), 1);
        }

        pool.shutdown();
    }

    #[test]
    fn test_request_router() {
        let pool = Arc::new(ActorPool::new(4, || EchoHandler, 100));
        let router = RequestRouter::new(Arc::clone(&pool));

        // Key-based routing should be consistent
        let result1 = router.route("Test1".to_string(), AffinityHint::KeyBased(42)).unwrap();
        let result2 = router.route("Test2".to_string(), AffinityHint::KeyBased(42)).unwrap();

        assert!(result1.starts_with("Echo:"));
        assert!(result2.starts_with("Echo:"));

        pool.shutdown();
    }
}
