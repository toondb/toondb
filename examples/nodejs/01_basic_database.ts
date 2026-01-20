/**
 * Basic SochDB Operations Example
 * 
 * Demonstrates fundamental key-value operations:
 * - Opening a database
 * - Put, Get, Delete operations
 * - Path-based hierarchical keys
 * - Prefix scan queries
 */

import { Database } from '@sochdb/sochdb';

async function main() {
  // Open or create a database
  const db = await Database.open('./example_db');
  console.log('✓ Database opened');

  try {
    // Basic key-value operations
    await db.put('greeting', 'Hello, SochDB!');
    console.log("✓ Key 'greeting' written");

    const value = await db.get('greeting');
    console.log(`✓ Read value: ${value?.toString()}`);

    // Path-based hierarchical keys
    await db.putPath(['users', 'alice', 'name'], 'Alice Smith');
    await db.putPath(['users', 'alice', 'email'], 'alice@example.com');
    await db.putPath(['users', 'bob', 'name'], 'Bob Jones');
    console.log('✓ Hierarchical data stored');

    // Read by path
    const aliceName = await db.getPath(['users', 'alice', 'name']);
    console.log(`✓ Alice's name: ${aliceName?.toString()}`);

    // Delete a key
    await db.delete('greeting');
    console.log("✓ Key 'greeting' deleted");

    // Verify deletion
    const deleted = await db.get('greeting');
    if (deleted === null) {
      console.log('✓ Key confirmed deleted');
    }

    // Query prefix scan
    const results = await db.query('users/')
      .limit(10)
      .execute();

    console.log('\n✓ Prefix scan results:');
    for (const [key, value] of results) {
      console.log(`  ${key} = ${value.toString()}`);
    }

  } finally {
    await db.close();
    console.log('\n✓ Database closed');
  }
}

main().catch(console.error);
