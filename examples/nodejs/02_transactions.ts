/**
 * Transaction Example
 * 
 * Demonstrates ACID transactions:
 * - Using withTransaction for automatic commit/rollback
 * - Read operations within transactions
 * - Error handling and rollback
 */

import { Database } from '@sochdb/sochdb';

async function main() {
  const db = await Database.open('./txn_example_db');
  console.log('✓ Database opened');

  try {
    // Automatic transaction with callback (recommended)
    await db.withTransaction(async (txn) => {
      await txn.put('accounts/alice/balance', '1000');
      await txn.put('accounts/bob/balance', '500');
      console.log('✓ Transaction: wrote initial balances');
    });
    console.log('✓ Transaction committed automatically');

    // Simulate a transfer
    await db.withTransaction(async (txn) => {
      // Read current balances
      const aliceData = await txn.get('accounts/alice/balance');
      const bobData = await txn.get('accounts/bob/balance');

      const aliceBalance = parseInt(aliceData?.toString() || '0');
      const bobBalance = parseInt(bobData?.toString() || '0');

      const transferAmount = 250;

      // Update balances atomically
      await txn.put('accounts/alice/balance', (aliceBalance - transferAmount).toString());
      await txn.put('accounts/bob/balance', (bobBalance + transferAmount).toString());

      console.log(`✓ Transfer: Alice -> Bob: $${transferAmount}`);
    });

    // Verify final balances
    const alice = await db.get('accounts/alice/balance');
    const bob = await db.get('accounts/bob/balance');

    console.log('\n✓ Final balances:');
    console.log(`  Alice: $${alice?.toString()}`);
    console.log(`  Bob: $${bob?.toString()}`);

    // Transaction rollback on error
    try {
      await db.withTransaction(async (txn) => {
        await txn.put('accounts/alice/balance', '9999');
        throw new Error('Simulated failure');
      });
    } catch (e) {
      console.log(`✓ Transaction rolled back: ${(e as Error).message}`);
    }

    // Verify balance unchanged after rollback
    const aliceAfter = await db.get('accounts/alice/balance');
    console.log(`✓ Alice's balance after rollback: $${aliceAfter?.toString()}`);

  } finally {
    await db.close();
  }
}

main().catch(console.error);
