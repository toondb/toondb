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

//! SQL Integration Tests
//!
//! Comprehensive test suite for SQL query execution with proper setup/teardown.
//! Tests cover all SQL statement types: SELECT, INSERT, UPDATE, DELETE, CREATE, DROP.

use sochdb_core::SochValue;
use sochdb_query::sql::{ExecutionResult, SqlError, SqlExecutor};

/// Test fixture providing setup and teardown for SQL tests
struct SqlTestFixture {
    executor: SqlExecutor,
}

impl SqlTestFixture {
    /// Create a new test fixture with empty database
    fn new() -> Self {
        Self {
            executor: SqlExecutor::new(),
        }
    }

    /// Setup a users table with sample data
    fn setup_users_table(&mut self) -> Result<(), SqlError> {
        self.executor.execute(
            "CREATE TABLE users (
            id INTEGER,
            name TEXT,
            email TEXT,
            age INTEGER,
            active INTEGER
        )",
        )?;

        self.executor.execute("INSERT INTO users (id, name, email, age, active) VALUES (1, 'Alice', 'alice@example.com', 30, 1)")?;
        self.executor.execute("INSERT INTO users (id, name, email, age, active) VALUES (2, 'Bob', 'bob@example.com', 25, 1)")?;
        self.executor.execute("INSERT INTO users (id, name, email, age, active) VALUES (3, 'Charlie', 'charlie@example.com', 35, 0)")?;
        self.executor.execute("INSERT INTO users (id, name, email, age, active) VALUES (4, 'Diana', 'diana@example.com', 28, 1)")?;
        self.executor.execute("INSERT INTO users (id, name, email, age, active) VALUES (5, 'Eve', 'eve@example.com', 32, 0)")?;

        Ok(())
    }

    /// Setup a products table with sample data
    fn setup_products_table(&mut self) -> Result<(), SqlError> {
        self.executor.execute(
            "CREATE TABLE products (
            id INTEGER,
            name TEXT,
            price REAL,
            stock INTEGER,
            category TEXT
        )",
        )?;

        self.executor.execute("INSERT INTO products (id, name, price, stock, category) VALUES (1, 'Laptop', 999.99, 50, 'Electronics')")?;
        self.executor.execute("INSERT INTO products (id, name, price, stock, category) VALUES (2, 'Mouse', 29.99, 200, 'Electronics')")?;
        self.executor.execute("INSERT INTO products (id, name, price, stock, category) VALUES (3, 'Keyboard', 79.99, 150, 'Electronics')")?;
        self.executor.execute("INSERT INTO products (id, name, price, stock, category) VALUES (4, 'Desk', 299.99, 30, 'Furniture')")?;
        self.executor.execute("INSERT INTO products (id, name, price, stock, category) VALUES (5, 'Chair', 199.99, 75, 'Furniture')")?;

        Ok(())
    }

    /// Setup an orders table with sample data
    #[allow(dead_code)]
    fn setup_orders_table(&mut self) -> Result<(), SqlError> {
        self.executor.execute(
            "CREATE TABLE orders (
            id INTEGER,
            user_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            total REAL
        )",
        )?;

        self.executor.execute("INSERT INTO orders (id, user_id, product_id, quantity, total) VALUES (1, 1, 1, 1, 999.99)")?;
        self.executor.execute("INSERT INTO orders (id, user_id, product_id, quantity, total) VALUES (2, 1, 2, 2, 59.98)")?;
        self.executor.execute("INSERT INTO orders (id, user_id, product_id, quantity, total) VALUES (3, 2, 3, 1, 79.99)")?;
        self.executor.execute("INSERT INTO orders (id, user_id, product_id, quantity, total) VALUES (4, 3, 4, 1, 299.99)")?;
        self.executor.execute("INSERT INTO orders (id, user_id, product_id, quantity, total) VALUES (5, 4, 5, 2, 399.98)")?;

        Ok(())
    }

    /// Teardown by dropping all tables
    fn teardown(&mut self) {
        let _ = self.executor.execute("DROP TABLE IF EXISTS users");
        let _ = self.executor.execute("DROP TABLE IF EXISTS products");
        let _ = self.executor.execute("DROP TABLE IF EXISTS orders");
    }

    /// Execute SQL and return result
    fn exec(&mut self, sql: &str) -> Result<ExecutionResult, SqlError> {
        self.executor.execute(sql)
    }

    /// Get row count from result
    fn row_count(&self, result: &ExecutionResult) -> usize {
        match result {
            ExecutionResult::Rows { rows, .. } => rows.len(),
            ExecutionResult::RowsAffected(n) => *n,
            ExecutionResult::Ok => 0,
        }
    }
}

impl Drop for SqlTestFixture {
    fn drop(&mut self) {
        self.teardown();
    }
}

// ============================================================================
// CREATE TABLE Tests
// ============================================================================

#[test]
fn test_create_table_basic() {
    let mut fixture = SqlTestFixture::new();

    let result = fixture.exec("CREATE TABLE test (id INTEGER, name TEXT)");
    assert!(result.is_ok(), "CREATE TABLE should succeed");
}

#[test]
fn test_create_table_if_not_exists() {
    let mut fixture = SqlTestFixture::new();

    fixture.exec("CREATE TABLE test (id INTEGER)").unwrap();

    // Should not error with IF NOT EXISTS
    let result = fixture.exec("CREATE TABLE IF NOT EXISTS test (id INTEGER)");
    assert!(result.is_ok());
}

#[test]
fn test_create_table_duplicate_error() {
    let mut fixture = SqlTestFixture::new();

    fixture.exec("CREATE TABLE test (id INTEGER)").unwrap();

    // Should error without IF NOT EXISTS
    let result = fixture.exec("CREATE TABLE test (id INTEGER)");
    assert!(result.is_err());
}

#[test]
fn test_create_table_with_multiple_columns() {
    let mut fixture = SqlTestFixture::new();

    let result = fixture.exec(
        "CREATE TABLE employees (
        id INTEGER,
        first_name TEXT,
        last_name TEXT,
        salary REAL,
        department TEXT,
        hire_date TEXT
    )",
    );
    assert!(result.is_ok());

    // Insert a row to verify columns work
    let insert = fixture.exec(
        "INSERT INTO employees (id, first_name, last_name, salary, department, hire_date) 
         VALUES (1, 'John', 'Doe', 75000.00, 'Engineering', '2024-01-15')",
    );
    assert!(insert.is_ok());
}

// ============================================================================
// INSERT Tests
// ============================================================================

#[test]
fn test_insert_basic() {
    let mut fixture = SqlTestFixture::new();
    fixture
        .exec("CREATE TABLE test (id INTEGER, name TEXT)")
        .unwrap();

    let result = fixture.exec("INSERT INTO test (id, name) VALUES (1, 'Alice')");
    assert!(result.is_ok());
    assert_eq!(fixture.row_count(&result.unwrap()), 1);
}

#[test]
fn test_insert_multiple_rows() {
    let mut fixture = SqlTestFixture::new();
    fixture
        .exec("CREATE TABLE test (id INTEGER, value TEXT)")
        .unwrap();

    for i in 1..=100 {
        let sql = format!("INSERT INTO test (id, value) VALUES ({}, 'value{}')", i, i);
        fixture.exec(&sql).unwrap();
    }

    let result = fixture.exec("SELECT * FROM test").unwrap();
    assert_eq!(fixture.row_count(&result), 100);
}

#[test]
fn test_insert_null_values() {
    let mut fixture = SqlTestFixture::new();
    fixture
        .exec("CREATE TABLE test (id INTEGER, optional TEXT)")
        .unwrap();

    let result = fixture.exec("INSERT INTO test (id, optional) VALUES (1, NULL)");
    assert!(result.is_ok());

    let select = fixture.exec("SELECT * FROM test").unwrap();
    if let ExecutionResult::Rows { rows, .. } = select {
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert!(matches!(row.get("optional"), Some(SochValue::Null)));
    }
}

#[test]
fn test_insert_with_expressions() {
    let mut fixture = SqlTestFixture::new();
    fixture
        .exec("CREATE TABLE test (id INTEGER, computed INTEGER)")
        .unwrap();

    // Insert with arithmetic expression
    let result = fixture.exec("INSERT INTO test (id, computed) VALUES (1, 10 + 5 * 2)");
    assert!(result.is_ok());

    let select = fixture.exec("SELECT * FROM test").unwrap();
    if let ExecutionResult::Rows { rows, .. } = select {
        let row = &rows[0];
        // 10 + 5 * 2 = 20 (multiplication has precedence)
        assert!(matches!(row.get("computed"), Some(SochValue::Int(20))));
    }
}

// ============================================================================
// SELECT Tests
// ============================================================================

#[test]
fn test_select_all() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec("SELECT * FROM users").unwrap();
    assert_eq!(fixture.row_count(&result), 5);
}

#[test]
fn test_select_specific_columns() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec("SELECT name, email FROM users").unwrap();
    if let ExecutionResult::Rows { columns, rows } = result {
        assert_eq!(columns.len(), 2);
        assert!(columns.contains(&"name".to_string()));
        assert!(columns.contains(&"email".to_string()));
        assert_eq!(rows.len(), 5);
    } else {
        panic!("Expected rows");
    }
}

#[test]
fn test_select_with_where_equals() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec("SELECT * FROM users WHERE id = 1").unwrap();
    if let ExecutionResult::Rows { rows, .. } = result {
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert!(matches!(row.get("name"), Some(SochValue::Text(s)) if s == "Alice"));
    } else {
        panic!("Expected rows");
    }
}

#[test]
fn test_select_with_where_greater_than() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec("SELECT * FROM users WHERE age > 30").unwrap();
    assert_eq!(fixture.row_count(&result), 2); // Charlie (35), Eve (32)
}

#[test]
fn test_select_with_where_less_than() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec("SELECT * FROM users WHERE age < 30").unwrap();
    assert_eq!(fixture.row_count(&result), 2); // Bob (25), Diana (28)
}

#[test]
fn test_select_with_where_and() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("SELECT * FROM users WHERE age > 25 AND active = 1")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 2); // Alice (30, active), Diana (28, active)
}

#[test]
fn test_select_with_where_or() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("SELECT * FROM users WHERE name = 'Alice' OR name = 'Bob'")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 2);
}

#[test]
fn test_select_with_where_between() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("SELECT * FROM users WHERE age BETWEEN 28 AND 32")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 3); // Alice (30), Diana (28), Eve (32)
}

#[test]
fn test_select_with_where_in() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("SELECT * FROM users WHERE id IN (1, 3, 5)")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 3);
}

#[test]
fn test_select_with_where_like() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("SELECT * FROM users WHERE email LIKE '%@example.com'")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 5); // All emails match
}

#[test]
fn test_select_with_order_by_asc() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("SELECT * FROM users ORDER BY age ASC")
        .unwrap();
    if let ExecutionResult::Rows { rows, .. } = result {
        let ages: Vec<i64> = rows
            .iter()
            .filter_map(|r| r.get("age"))
            .filter_map(|v| {
                if let SochValue::Int(n) = v {
                    Some(*n)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(ages, vec![25, 28, 30, 32, 35]);
    }
}

#[test]
fn test_select_with_order_by_desc() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("SELECT * FROM users ORDER BY age DESC")
        .unwrap();
    if let ExecutionResult::Rows { rows, .. } = result {
        let ages: Vec<i64> = rows
            .iter()
            .filter_map(|r| r.get("age"))
            .filter_map(|v| {
                if let SochValue::Int(n) = v {
                    Some(*n)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(ages, vec![35, 32, 30, 28, 25]);
    }
}

#[test]
fn test_select_with_limit() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec("SELECT * FROM users LIMIT 3").unwrap();
    assert_eq!(fixture.row_count(&result), 3);
}

#[test]
fn test_select_with_offset() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("SELECT * FROM users LIMIT 3 OFFSET 2")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 3);
}

#[test]
fn test_select_with_functions() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    // Test UPPER function
    let result = fixture
        .exec("SELECT UPPER(name) FROM users WHERE id = 1")
        .unwrap();
    if let ExecutionResult::Rows { rows, .. } = result {
        assert!(
            rows[0]
                .values()
                .any(|v| matches!(v, SochValue::Text(s) if s == "ALICE"))
        );
    }
}

#[test]
fn test_select_with_lower_function() {
    let mut fixture = SqlTestFixture::new();
    fixture.exec("CREATE TABLE test (name TEXT)").unwrap();
    fixture
        .exec("INSERT INTO test (name) VALUES ('HELLO WORLD')")
        .unwrap();

    let result = fixture.exec("SELECT LOWER(name) FROM test").unwrap();
    if let ExecutionResult::Rows { rows, .. } = result {
        assert!(
            rows[0]
                .values()
                .any(|v| matches!(v, SochValue::Text(s) if s == "hello world"))
        );
    }
}

#[test]
fn test_select_with_length_function() {
    let mut fixture = SqlTestFixture::new();
    fixture.exec("CREATE TABLE test (name TEXT)").unwrap();
    fixture
        .exec("INSERT INTO test (name) VALUES ('Hello')")
        .unwrap();

    let result = fixture.exec("SELECT LENGTH(name) FROM test").unwrap();
    if let ExecutionResult::Rows { rows, .. } = result {
        assert!(rows[0].values().any(|v| matches!(v, SochValue::Int(5))));
    }
}

#[test]
fn test_select_with_coalesce() {
    let mut fixture = SqlTestFixture::new();
    fixture
        .exec("CREATE TABLE test (a INTEGER, b INTEGER)")
        .unwrap();
    fixture
        .exec("INSERT INTO test (a, b) VALUES (NULL, 5)")
        .unwrap();

    let result = fixture.exec("SELECT COALESCE(a, b) FROM test").unwrap();
    if let ExecutionResult::Rows { rows, .. } = result {
        assert!(rows[0].values().any(|v| matches!(v, SochValue::Int(5))));
    }
}

#[test]
fn test_select_with_arithmetic() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_products_table().unwrap();

    let result = fixture
        .exec("SELECT price * 1.1 FROM products WHERE id = 1")
        .unwrap();
    if let ExecutionResult::Rows { rows, .. } = result {
        // Laptop price 999.99 * 1.1 = 1099.989
        let value = rows[0].values().next().unwrap();
        if let SochValue::Float(f) = value {
            assert!((*f - 1099.989).abs() < 0.01);
        }
    }
}

// ============================================================================
// UPDATE Tests
// ============================================================================

#[test]
fn test_update_single_row() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("UPDATE users SET age = 31 WHERE id = 1")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 1);

    let select = fixture.exec("SELECT age FROM users WHERE id = 1").unwrap();
    if let ExecutionResult::Rows { rows, .. } = select {
        assert!(matches!(rows[0].get("age"), Some(SochValue::Int(31))));
    }
}

#[test]
fn test_update_multiple_columns() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("UPDATE users SET name = 'Alicia', age = 31 WHERE id = 1")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 1);

    let select = fixture.exec("SELECT * FROM users WHERE id = 1").unwrap();
    if let ExecutionResult::Rows { rows, .. } = select {
        assert!(matches!(rows[0].get("name"), Some(SochValue::Text(s)) if s == "Alicia"));
        assert!(matches!(rows[0].get("age"), Some(SochValue::Int(31))));
    }
}

#[test]
fn test_update_multiple_rows() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("UPDATE users SET active = 1 WHERE active = 0")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 2); // Charlie and Eve

    let select = fixture
        .exec("SELECT * FROM users WHERE active = 0")
        .unwrap();
    assert_eq!(fixture.row_count(&select), 0); // No more inactive users
}

#[test]
fn test_update_all_rows() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec("UPDATE users SET active = 0").unwrap();
    assert_eq!(fixture.row_count(&result), 5);
}

#[test]
fn test_update_with_expression() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("UPDATE users SET age = age + 1 WHERE id = 1")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 1);

    let select = fixture.exec("SELECT age FROM users WHERE id = 1").unwrap();
    if let ExecutionResult::Rows { rows, .. } = select {
        assert!(matches!(rows[0].get("age"), Some(SochValue::Int(31)))); // Was 30
    }
}

// ============================================================================
// DELETE Tests
// ============================================================================

#[test]
fn test_delete_single_row() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec("DELETE FROM users WHERE id = 1").unwrap();
    assert_eq!(fixture.row_count(&result), 1);

    let select = fixture.exec("SELECT * FROM users").unwrap();
    assert_eq!(fixture.row_count(&select), 4);
}

#[test]
fn test_delete_multiple_rows() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec("DELETE FROM users WHERE active = 0").unwrap();
    assert_eq!(fixture.row_count(&result), 2); // Charlie and Eve

    let select = fixture.exec("SELECT * FROM users").unwrap();
    assert_eq!(fixture.row_count(&select), 3);
}

#[test]
fn test_delete_all_rows() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec("DELETE FROM users").unwrap();
    assert_eq!(fixture.row_count(&result), 5);

    let select = fixture.exec("SELECT * FROM users").unwrap();
    assert_eq!(fixture.row_count(&select), 0);
}

#[test]
fn test_delete_with_complex_where() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture
        .exec("DELETE FROM users WHERE age > 30 OR active = 0")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 2); // Charlie (35, inactive=0), Eve (32, inactive=0)

    // Remaining: Alice (30, active), Bob (25, active), Diana (28, active)
}

// ============================================================================
// DROP TABLE Tests
// ============================================================================

#[test]
fn test_drop_table() {
    let mut fixture = SqlTestFixture::new();
    fixture.exec("CREATE TABLE test (id INTEGER)").unwrap();

    let result = fixture.exec("DROP TABLE test");
    assert!(result.is_ok());

    // Table should no longer exist
    let select = fixture.exec("SELECT * FROM test");
    assert!(select.is_err());
}

#[test]
fn test_drop_table_if_exists() {
    let mut fixture = SqlTestFixture::new();

    // Should not error even if table doesn't exist
    let result = fixture.exec("DROP TABLE IF EXISTS nonexistent");
    assert!(result.is_ok());
}

#[test]
fn test_drop_table_error_if_not_exists() {
    let mut fixture = SqlTestFixture::new();

    let result = fixture.exec("DROP TABLE nonexistent");
    assert!(result.is_err());
}

// ============================================================================
// Complex Query Tests
// ============================================================================

#[test]
fn test_complex_query_with_multiple_conditions() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec(
        "SELECT name, age FROM users WHERE (age >= 28 AND age <= 32) AND active = 1 ORDER BY age DESC"
    ).unwrap();

    if let ExecutionResult::Rows { rows, .. } = result {
        // Alice (30, active), Diana (28, active)
        assert_eq!(rows.len(), 2);

        let ages: Vec<i64> = rows
            .iter()
            .filter_map(|r| r.get("age"))
            .filter_map(|v| {
                if let SochValue::Int(n) = v {
                    Some(*n)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(ages, vec![30, 28]); // Ordered DESC
    }
}

#[test]
fn test_workflow_crud_operations() {
    let mut fixture = SqlTestFixture::new();

    // Create
    fixture
        .exec("CREATE TABLE items (id INTEGER, name TEXT, quantity INTEGER)")
        .unwrap();

    // Insert
    fixture
        .exec("INSERT INTO items (id, name, quantity) VALUES (1, 'Widget', 100)")
        .unwrap();
    fixture
        .exec("INSERT INTO items (id, name, quantity) VALUES (2, 'Gadget', 50)")
        .unwrap();

    // Read
    let result = fixture.exec("SELECT * FROM items").unwrap();
    assert_eq!(fixture.row_count(&result), 2);

    // Update
    fixture
        .exec("UPDATE items SET quantity = 75 WHERE id = 2")
        .unwrap();
    let result = fixture
        .exec("SELECT quantity FROM items WHERE id = 2")
        .unwrap();
    if let ExecutionResult::Rows { rows, .. } = result {
        assert!(matches!(rows[0].get("quantity"), Some(SochValue::Int(75))));
    }

    // Delete
    fixture.exec("DELETE FROM items WHERE id = 1").unwrap();
    let result = fixture.exec("SELECT * FROM items").unwrap();
    assert_eq!(fixture.row_count(&result), 1);

    // Drop
    fixture.exec("DROP TABLE items").unwrap();
}

#[test]
fn test_products_price_filtering() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_products_table().unwrap();

    // Find products under $100
    let result = fixture
        .exec("SELECT * FROM products WHERE price < 100.0")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 2); // Mouse (29.99), Keyboard (79.99)

    // Find electronics
    let result = fixture
        .exec("SELECT * FROM products WHERE category = 'Electronics'")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 3);
}

#[test]
fn test_null_handling() {
    let mut fixture = SqlTestFixture::new();
    fixture
        .exec("CREATE TABLE nullable (id INTEGER, value TEXT)")
        .unwrap();
    fixture
        .exec("INSERT INTO nullable (id, value) VALUES (1, 'not null')")
        .unwrap();
    fixture
        .exec("INSERT INTO nullable (id, value) VALUES (2, NULL)")
        .unwrap();

    // IS NULL
    let result = fixture
        .exec("SELECT * FROM nullable WHERE value IS NULL")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 1);

    // IS NOT NULL
    let result = fixture
        .exec("SELECT * FROM nullable WHERE value IS NOT NULL")
        .unwrap();
    assert_eq!(fixture.row_count(&result), 1);
}

#[test]
fn test_case_expression() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec(
        "SELECT name, CASE WHEN age < 30 THEN 'young' WHEN age < 35 THEN 'middle' ELSE 'senior' END FROM users"
    ).unwrap();

    assert_eq!(fixture.row_count(&result), 5);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_error_select_nonexistent_table() {
    let mut fixture = SqlTestFixture::new();

    let result = fixture.exec("SELECT * FROM nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_error_insert_nonexistent_table() {
    let mut fixture = SqlTestFixture::new();

    let result = fixture.exec("INSERT INTO nonexistent (id) VALUES (1)");
    assert!(result.is_err());
}

#[test]
fn test_error_update_nonexistent_table() {
    let mut fixture = SqlTestFixture::new();

    let result = fixture.exec("UPDATE nonexistent SET id = 1");
    assert!(result.is_err());
}

#[test]
fn test_error_delete_nonexistent_table() {
    let mut fixture = SqlTestFixture::new();

    let result = fixture.exec("DELETE FROM nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_syntax_error_handling() {
    let mut fixture = SqlTestFixture::new();

    let result = fixture.exec("SELEKT * FROM users"); // Typo
    assert!(result.is_err());
}

// ============================================================================
// Performance/Stress Tests
// ============================================================================

#[test]
fn test_bulk_insert_1000_rows() {
    let mut fixture = SqlTestFixture::new();
    fixture
        .exec("CREATE TABLE bulk (id INTEGER, data TEXT)")
        .unwrap();

    for i in 0..1000 {
        let sql = format!("INSERT INTO bulk (id, data) VALUES ({}, 'data_{}')", i, i);
        fixture.exec(&sql).unwrap();
    }

    let result = fixture.exec("SELECT * FROM bulk").unwrap();
    assert_eq!(fixture.row_count(&result), 1000);
}

#[test]
fn test_select_with_many_conditions() {
    let mut fixture = SqlTestFixture::new();
    fixture.setup_users_table().unwrap();

    let result = fixture.exec(
        "SELECT * FROM users WHERE id >= 1 AND id <= 5 AND age >= 25 AND age <= 35 AND active = 1"
    ).unwrap();

    // Should return Alice (30), Bob (25), Diana (28)
    assert_eq!(fixture.row_count(&result), 3);
}
