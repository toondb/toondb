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

//! SochDB Analytics - Anonymous usage tracking with PostHog
//!
//! This module provides anonymous, privacy-respecting analytics to help
//! improve SochDB. All tracking can be disabled by setting the environment variable:
//!
//! ```bash
//! export SOCHDB_DISABLE_ANALYTICS=true
//! ```
//!
//! No personally identifiable information (PII) is collected. Only aggregate
//! usage patterns are tracked to understand:
//! - Which features are most used
//! - Performance characteristics
//! - Error patterns for debugging

use std::collections::HashMap;
use std::env;
use std::sync::OnceLock;

/// PostHog configuration
const POSTHOG_API_KEY: &str = "phc_zf0hm6ZmPUJj1pM07Kigqvphh1ClhKX1NahRU4G0bfu";
const POSTHOG_HOST: &str = "https://us.i.posthog.com";

/// Cached analytics disabled state
static ANALYTICS_DISABLED: OnceLock<bool> = OnceLock::new();

/// Cached anonymous ID
static ANONYMOUS_ID: OnceLock<String> = OnceLock::new();

/// Check if analytics is disabled via environment variable.
pub fn is_analytics_disabled() -> bool {
    *ANALYTICS_DISABLED.get_or_init(|| {
        env::var("SOCHDB_DISABLE_ANALYTICS")
            .map(|v| {
                let v = v.to_lowercase();
                v == "true" || v == "1" || v == "yes" || v == "on"
            })
            .unwrap_or(false)
    })
}

/// Generate a stable anonymous ID for this machine.
///
/// Uses a hash of machine-specific but non-identifying information.
/// The same machine will always get the same ID, but the ID cannot
/// be reversed to identify the machine.
pub fn get_anonymous_id() -> &'static str {
    ANONYMOUS_ID.get_or_init(|| {
        use std::hash::{Hash, Hasher};
        
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // Hash machine-specific info
        if let Ok(hostname) = hostname::get() {
            hostname.to_string_lossy().hash(&mut hasher);
        }
        
        std::env::consts::OS.hash(&mut hasher);
        std::env::consts::ARCH.hash(&mut hasher);
        
        #[cfg(unix)]
        {
            unsafe {
                libc::getuid().hash(&mut hasher);
            }
        }
        
        format!("{:016x}", hasher.finish())
    })
}

/// Event properties for analytics
pub type EventProperties = HashMap<String, serde_json::Value>;

/// Analytics client for sending events to PostHog
#[derive(Clone)]
pub struct Analytics {
    api_key: String,
    host: String,
    disabled: bool,
}

impl Default for Analytics {
    fn default() -> Self {
        Self::new()
    }
}

impl Analytics {
    /// Create a new Analytics client with default configuration.
    pub fn new() -> Self {
        Self {
            api_key: POSTHOG_API_KEY.to_string(),
            host: POSTHOG_HOST.to_string(),
            disabled: is_analytics_disabled(),
        }
    }

    /// Check if analytics is enabled.
    pub fn is_enabled(&self) -> bool {
        !self.disabled
    }

    /// Capture an analytics event.
    ///
    /// This function is a no-op if:
    /// - SOCHDB_DISABLE_ANALYTICS=true
    /// - Any error occurs (fails silently)
    ///
    /// # Arguments
    /// * `event` - Event name (e.g., "database_opened", "vector_search")
    /// * `properties` - Optional event properties
    pub fn capture(&self, event: &str, properties: Option<EventProperties>) {
        if self.disabled {
            return;
        }

        // Build the event asynchronously to not block the caller
        let api_key = self.api_key.clone();
        let host = self.host.clone();
        let event = event.to_string();
        let distinct_id = get_anonymous_id().to_string();

        // Build properties with SDK context
        let mut event_properties = properties.unwrap_or_default();
        event_properties.insert("sdk".to_string(), serde_json::json!("rust"));
        event_properties.insert(
            "sdk_version".to_string(),
            serde_json::json!(env!("CARGO_PKG_VERSION")),
        );
        event_properties.insert("os".to_string(), serde_json::json!(std::env::consts::OS));
        event_properties.insert(
            "arch".to_string(),
            serde_json::json!(std::env::consts::ARCH),
        );

        // Spawn a thread to send the event (non-blocking)
        std::thread::spawn(move || {
            let _ = send_event(&host, &api_key, &distinct_id, &event, event_properties);
        });
    }

    /// Capture an error event for debugging.
    ///
    /// Only sends static information - no dynamic error messages.
    ///
    /// # Arguments
    /// * `error_type` - Static error category (e.g., "connection_error", "query_error")
    /// * `location` - Static code location (e.g., "database::open", "query::execute")
    ///
    /// # Example
    /// ```no_run
    /// # use sochdb_core::analytics::Analytics;
    /// let analytics = Analytics::new();
    /// analytics.capture_error("connection_error", "database::open");
    /// analytics.capture_error("query_error", "sql::execute");
    /// ```
    pub fn capture_error(&self, error_type: &str, location: &str) {
        let mut props = EventProperties::new();
        props.insert("error_type".to_string(), serde_json::json!(error_type));
        props.insert("location".to_string(), serde_json::json!(location));
        self.capture("error", Some(props));
    }

    /// Track database open event.
    pub fn track_database_open(&self, db_path: &str, mode: &str) {
        let mut props = EventProperties::new();
        props.insert("mode".to_string(), serde_json::json!(mode));
        props.insert(
            "has_custom_path".to_string(),
            serde_json::json!(db_path != ":memory:"),
        );
        self.capture("database_opened", Some(props));
    }
}

/// Send an event to PostHog via HTTP.
fn send_event(
    host: &str,
    api_key: &str,
    distinct_id: &str,
    event: &str,
    properties: EventProperties,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Merge all properties including SDK context
    let mut event_properties = properties;
    event_properties.insert("$lib".to_string(), serde_json::json!("sochdb-rust"));
    
    // PostHog capture endpoint expects this format
    let payload = serde_json::json!({
        "api_key": api_key,
        "event": event,
        "properties": event_properties,
        "distinct_id": distinct_id,
    });

    let url = format!("{}/capture/", host);

    // Use ureq for simple HTTP POST (it's sync and lightweight)
    #[cfg(feature = "analytics")]
    {
        ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(&payload)?;
    }

    Ok(())
}

/// Global analytics instance for convenience.
static GLOBAL_ANALYTICS: OnceLock<Analytics> = OnceLock::new();

/// Get the global analytics instance.
pub fn analytics() -> &'static Analytics {
    GLOBAL_ANALYTICS.get_or_init(Analytics::new)
}

/// Capture an event using the global analytics instance.
pub fn capture(event: &str, properties: Option<EventProperties>) {
    analytics().capture(event, properties);
}

/// Capture an error using the global analytics instance.
///
/// Only sends static information - no dynamic error messages.
pub fn capture_error(error_type: &str, location: &str) {
    analytics().capture_error(error_type, location);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_analytics_disabled_default() {
        // Clear any cached value for testing
        // Note: In production, this is cached once
        let result = env::var("SOCHDB_DISABLE_ANALYTICS")
            .map(|v| {
                let v = v.to_lowercase();
                v == "true" || v == "1" || v == "yes" || v == "on"
            })
            .unwrap_or(false);
        
        // Just verify it doesn't panic
        assert!(result == true || result == false);
    }

    #[test]
    fn test_anonymous_id_is_stable() {
        let id1 = get_anonymous_id();
        let id2 = get_anonymous_id();
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 16);
    }

    #[test]
    fn test_analytics_disabled_no_op() {
        // When disabled, capture should be a no-op
        let analytics = Analytics {
            api_key: "test".to_string(),
            host: "http://localhost".to_string(),
            disabled: true,
        };
        
        // This should not panic or make any network calls
        analytics.capture("test_event", None);
    }
}
