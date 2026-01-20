/**
 * SochDB SQL Query Examples
 * 
 * Demonstrates SQL support in SochDB:
 * - CREATE TABLE, INSERT, UPDATE, DELETE
 * - SELECT with WHERE, ORDER BY, LIMIT
 * - Transactions with SQL
 * - Schema management
 */

import { Database } from '@sochdb/sochdb';
import * as fs from 'fs';
import * as path from 'path';

async function createTables(db: Database): Promise<void> {
    console.log('\n📝 Creating Tables with SQL');
    console.log('='.repeat(60));

    // Create users table
    await db.execute(`
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            age INTEGER,
            created_at TEXT
        )
    `);
    console.log('✓ Created \'users\' table');

    // Create posts table
    await db.execute(`
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            title TEXT NOT NULL,
            content TEXT,
            likes INTEGER DEFAULT 0,
            published_at TEXT
        )
    `);
    console.log('✓ Created \'posts\' table');
}

async function insertData(db: Database): Promise<void> {
    console.log('\n📥 Inserting Data with SQL');
    console.log('='.repeat(60));

    // Insert users
    const users = [
        [1, 'Alice', 'alice@example.com', 30, '2024-01-01'],
        [2, 'Bob', 'bob@example.com', 25, '2024-01-02'],
        [3, 'Charlie', 'charlie@example.com', 35, '2024-01-03'],
        [4, 'Diana', 'diana@example.com', 28, '2024-01-04'],
    ];

    for (const [id, name, email, age, createdAt] of users) {
        await db.execute(`
            INSERT INTO users (id, name, email, age, created_at)
            VALUES (${id}, '${name}', '${email}', ${age}, '${createdAt}')
        `);
        console.log(`  ✓ Inserted user: ${name}`);
    }

    // Insert posts
    const posts = [
        [1, 1, 'First Post', 'Hello World!', 10, '2024-01-05'],
        [2, 1, 'Second Post', 'SochDB is awesome', 25, '2024-01-06'],
        [3, 2, 'Bob\'s Thoughts', 'SQL queries are easy', 15, '2024-01-07'],
        [4, 3, 'Charlie\'s Guide', 'Database tips', 30, '2024-01-08'],
        [5, 3, 'Advanced Topics', 'Performance tuning', 50, '2024-01-09'],
    ];

    for (const [id, userId, title, content, likes, publishedAt] of posts) {
        await db.execute(`
            INSERT INTO posts (id, user_id, title, content, likes, published_at)
            VALUES (${id}, ${userId}, '${title}', '${content}', ${likes}, '${publishedAt}')
        `);
        console.log(`  ✓ Inserted post: ${title}`);
    }
}

async function selectQueries(db: Database): Promise<void> {
    console.log('\n🔍 Running SELECT Queries');
    console.log('='.repeat(60));

    // Simple SELECT
    console.log('\n1. Select all users:');
    let result = await db.execute('SELECT * FROM users');
    console.log(`   Found ${result.rows.length} users`);
    for (const row of result.rows) {
        console.log(`   - ${row.name} (${row.email})`);
    }

    // SELECT with WHERE clause
    console.log('\n2. Users older than 28:');
    result = await db.execute('SELECT name, age FROM users WHERE age > 28');
    for (const row of result.rows) {
        console.log(`   - ${row.name}: ${row.age} years old`);
    }

    // SELECT with ORDER BY
    console.log('\n3. Posts ordered by likes (descending):');
    result = await db.execute('SELECT title, likes FROM posts ORDER BY likes DESC');
    for (const row of result.rows) {
        console.log(`   - ${row.title}: ${row.likes} likes`);
    }

    // SELECT with LIMIT
    console.log('\n4. Top 3 most liked posts:');
    result = await db.execute('SELECT title, likes FROM posts ORDER BY likes DESC LIMIT 3');
    for (const row of result.rows) {
        console.log(`   - ${row.title}: ${row.likes} likes`);
    }

    // Aggregate functions
    console.log('\n5. Count total posts:');
    result = await db.execute('SELECT COUNT(*) as total FROM posts');
    console.log(`   Total posts: ${result.rows[0]?.total || result.rows.length}`);
}

async function updateOperations(db: Database): Promise<void> {
    console.log('\n✏️  UPDATE Operations');
    console.log('='.repeat(60));

    // Update single row
    console.log('\n1. Update Alice\'s age:');
    await db.execute('UPDATE users SET age = 31 WHERE name = \'Alice\'');
    const result = await db.execute('SELECT name, age FROM users WHERE name = \'Alice\'');
    console.log(`   Alice's new age: ${result.rows[0].age}`);

    // Update multiple rows
    console.log('\n2. Increment likes on all posts by user_id = 1:');
    await db.execute('UPDATE posts SET likes = likes + 5 WHERE user_id = 1');
    const posts = await db.execute('SELECT title, likes FROM posts WHERE user_id = 1');
    for (const row of posts.rows) {
        console.log(`   - ${row.title}: ${row.likes} likes`);
    }
}

async function deleteOperations(db: Database): Promise<void> {
    console.log('\n🗑️  DELETE Operations');
    console.log('='.repeat(60));

    // Count before delete
    let result = await db.execute('SELECT COUNT(*) as total FROM posts');
    const beforeCount = result.rows.length;
    console.log(`Posts before delete: ${beforeCount}`);

    // Delete specific post
    await db.execute('DELETE FROM posts WHERE id = 5');
    console.log('✓ Deleted post with id = 5');

    // Count after delete
    result = await db.execute('SELECT COUNT(*) as total FROM posts');
    const afterCount = result.rows.length;
    console.log(`Posts after delete: ${afterCount}`);
}

async function transactionsWithSql(db: Database): Promise<void> {
    console.log('\n💳 SQL in Transactions');
    console.log('='.repeat(60));

    try {
        // Start transaction
        const txn = await db.beginTransaction();

        // Execute SQL within transaction
        await db.execute('INSERT INTO users (id, name, email, age) VALUES (5, \'Eve\', \'eve@example.com\', 26)');
        await db.execute('INSERT INTO posts (id, user_id, title, content) VALUES (6, 5, \'Eve Post\', \'My first post\')');

        // Commit transaction
        await txn.commit();
        console.log('✓ Transaction committed successfully');

        // Verify data
        const result = await db.execute('SELECT name FROM users WHERE id = 5');
        if (result.rows.length > 0) {
            console.log(`  New user: ${result.rows[0].name}`);
        }
    } catch (error) {
        console.error(`✗ Transaction failed: ${error}`);
    }
}

async function complexQueries(db: Database): Promise<void> {
    console.log('\n🎯 Complex Queries');
    console.log('='.repeat(60));

    // SELECT with multiple conditions
    console.log('\n1. Users aged 25-30:');
    let result = await db.execute(`
        SELECT name, age, email 
        FROM users 
        WHERE age >= 25 AND age <= 30
        ORDER BY age
    `);
    for (const row of result.rows) {
        console.log(`   - ${row.name}: ${row.age} years (${row.email})`);
    }

    // SELECT with LIKE
    console.log('\n2. Posts with \'Post\' in title:');
    result = await db.execute(`
        SELECT title, likes 
        FROM posts 
        WHERE title LIKE '%Post%'
    `);
    for (const row of result.rows) {
        console.log(`   - ${row.title}: ${row.likes} likes`);
    }
}

async function main() {
    console.log('='.repeat(60));
    console.log('SochDB SQL Query Examples');
    console.log('='.repeat(60));

    // Open database
    const dbPath = './demo_sql_db_ts';
    console.log(`\n📂 Opening database: ${dbPath}`);

    // Clean up existing database
    if (fs.existsSync(dbPath)) {
        fs.rmSync(dbPath, { recursive: true, force: true });
    }

    const db = new Database(dbPath);
    console.log('✓ Database opened successfully');

    try {
        // Run demonstrations
        await createTables(db);
        await insertData(db);
        await selectQueries(db);
        await updateOperations(db);
        await deleteOperations(db);
        await transactionsWithSql(db);
        await complexQueries(db);

        console.log('\n' + '='.repeat(60));
        console.log('✓ All SQL examples completed successfully!');
        console.log('='.repeat(60));
    } catch (error) {
        console.error(`\n✗ Error: ${error}`);
        console.error(error);
    } finally {
        // Close database
        await db.close();
        console.log('\n✓ Database closed');
    }
}

// Run if executed directly
if (require.main === module) {
    main().catch(console.error);
}

export { main };
