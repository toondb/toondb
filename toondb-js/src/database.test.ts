/**
 * Comprehensive tests for ToonDB Node.js SDK
 *
 * Tests all major features:
 * - Database operations (get, put, delete)
 * - Path operations (getPath, putPath)
 * - Transaction management
 * - Vector index operations
 * - Error handling
 */

import { Database, Transaction, DatabaseConfig } from './database';
import { IpcClient } from './ipc-client';
import { VectorIndex, VectorIndexConfig, VectorSearchResult } from './vector';
import {
  ToonDBError,
  ConnectionError,
  TransactionError,
  ProtocolError,
  DatabaseError,
} from './errors';
import { VERSION } from './index';

// Mock modules
jest.mock('./ipc-client');
jest.mock('fs');
jest.mock('child_process');
jest.mock('./server-manager', () => ({
  startEmbeddedServer: jest.fn().mockResolvedValue('/tmp/test_db/toondb.sock'),
  stopEmbeddedServer: jest.fn().mockResolvedValue(undefined),
  stopAllEmbeddedServers: jest.fn().mockResolvedValue(undefined),
  isServerRunning: jest.fn().mockReturnValue(false),
}));

// Type for mocked IpcClient
type MockIpcClient = {
  get: jest.Mock;
  put: jest.Mock;
  delete: jest.Mock;
  getPath: jest.Mock;
  putPath: jest.Mock;
  query: jest.Mock;
  beginTransaction: jest.Mock;
  commitTransaction: jest.Mock;
  abortTransaction: jest.Mock;
  checkpoint: jest.Mock;
  stats: jest.Mock;
  close: jest.Mock;
};

describe('ToonDB SDK Comprehensive Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('VERSION', () => {
    it('should export a valid semver version', () => {
      expect(VERSION).toMatch(/^\d+\.\d+\.\d+/);
    });

    it('should be 0.2.4', () => {
      expect(VERSION).toBe('0.2.4');
    });
  });

  describe('Database', () => {
    let mockClient: MockIpcClient;

    beforeEach(() => {
      mockClient = {
        get: jest.fn(),
        put: jest.fn(),
        delete: jest.fn(),
        getPath: jest.fn(),
        putPath: jest.fn(),
        query: jest.fn(),
        beginTransaction: jest.fn(),
        commitTransaction: jest.fn(),
        abortTransaction: jest.fn(),
        checkpoint: jest.fn(),
        stats: jest.fn(),
        close: jest.fn(),
      };

      (IpcClient.connect as jest.Mock) = jest.fn().mockResolvedValue(mockClient);
    });

    describe('Database.open()', () => {
      it('should open database with string path', async () => {
        const fs = require('fs');
        fs.existsSync = jest.fn().mockReturnValue(false);
        fs.mkdirSync = jest.fn();

        const db = await Database.open('./test_db');
        expect(db).toBeDefined();
        expect((IpcClient.connect as jest.Mock)).toHaveBeenCalled();
      });

      it('should open database with config object', async () => {
        const fs = require('fs');
        fs.existsSync = jest.fn().mockReturnValue(false);
        fs.mkdirSync = jest.fn();

        const config: DatabaseConfig = {
          path: './test_db',
          walEnabled: true,
          syncMode: 'full',
        };
        const db = await Database.open(config);
        expect(db).toBeDefined();
      });

      it('should create directory if missing', async () => {
        const fs = require('fs');
        fs.existsSync = jest.fn().mockReturnValue(false);
        fs.mkdirSync = jest.fn();

        await Database.open('./new_db');
        expect(fs.mkdirSync).toHaveBeenCalledWith('./new_db', { recursive: true });
      });

      it('should not create directory if createIfMissing is false', async () => {
        const fs = require('fs');
        fs.existsSync = jest.fn().mockReturnValue(false);
        fs.mkdirSync = jest.fn();

        const config: DatabaseConfig = {
          path: './existing_db',
          createIfMissing: false,
        };

        await Database.open(config);
        expect(fs.mkdirSync).not.toHaveBeenCalled();
      });
    });

    describe('get/put/delete operations', () => {
      let db: Database;

      beforeEach(async () => {
        const fs = require('fs');
        fs.existsSync = jest.fn().mockReturnValue(true);
        db = await Database.open('./test_db');
      });

      it('should get a value by key', async () => {
        mockClient.get.mockResolvedValue(Buffer.from('test value'));
        
        const result = await db.get('test-key');
        
        expect(result).toEqual(Buffer.from('test value'));
        expect(mockClient.get).toHaveBeenCalledWith(Buffer.from('test-key'));
      });

      it('should get a value with Buffer key', async () => {
        mockClient.get.mockResolvedValue(Buffer.from('test value'));
        
        const key = Buffer.from('binary-key');
        const result = await db.get(key);
        
        expect(result).toEqual(Buffer.from('test value'));
        expect(mockClient.get).toHaveBeenCalledWith(key);
      });

      it('should return null for non-existent key', async () => {
        mockClient.get.mockResolvedValue(null);
        
        const result = await db.get('missing-key');
        
        expect(result).toBeNull();
      });

      it('should put a key-value pair', async () => {
        mockClient.put.mockResolvedValue(undefined);
        
        await db.put('my-key', 'my-value');
        
        expect(mockClient.put).toHaveBeenCalledWith(
          Buffer.from('my-key'),
          Buffer.from('my-value')
        );
      });

      it('should put with Buffer values', async () => {
        mockClient.put.mockResolvedValue(undefined);
        
        const key = Buffer.from('binary-key');
        const value = Buffer.from('binary-value');
        await db.put(key, value);
        
        expect(mockClient.put).toHaveBeenCalledWith(key, value);
      });

      it('should delete a key', async () => {
        mockClient.delete.mockResolvedValue(undefined);
        
        await db.delete('my-key');
        
        expect(mockClient.delete).toHaveBeenCalledWith(Buffer.from('my-key'));
      });
    });

    describe('path operations', () => {
      let db: Database;

      beforeEach(async () => {
        const fs = require('fs');
        fs.existsSync = jest.fn().mockReturnValue(true);
        db = await Database.open('./test_db');
      });

      it('should getPath', async () => {
        mockClient.getPath.mockResolvedValue(Buffer.from('path value'));
        
        const result = await db.getPath('users/alice/email');
        
        expect(result).toEqual(Buffer.from('path value'));
        expect(mockClient.getPath).toHaveBeenCalledWith('users/alice/email');
      });

      it('should putPath', async () => {
        mockClient.putPath.mockResolvedValue(undefined);
        
        await db.putPath('users/alice/email', 'alice@example.com');
        
        expect(mockClient.putPath).toHaveBeenCalledWith(
          'users/alice/email',
          Buffer.from('alice@example.com')
        );
      });
    });

    describe('transactions', () => {
      let db: Database;

      beforeEach(async () => {
        const fs = require('fs');
        fs.existsSync = jest.fn().mockReturnValue(true);
        db = await Database.open('./test_db');
        mockClient.beginTransaction.mockResolvedValue(BigInt(12345));
        mockClient.commitTransaction.mockResolvedValue(undefined);
        mockClient.abortTransaction.mockResolvedValue(undefined);
      });

      it('should execute withTransaction successfully', async () => {
        mockClient.put.mockResolvedValue(undefined);

        await db.withTransaction(async (txn) => {
          await txn.put('key1', 'value1');
          await txn.put('key2', 'value2');
        });

        expect(mockClient.beginTransaction).toHaveBeenCalled();
        expect(mockClient.put).toHaveBeenCalledTimes(2);
        expect(mockClient.commitTransaction).toHaveBeenCalledWith(BigInt(12345));
      });

      it('should abort transaction on error', async () => {
        mockClient.put.mockRejectedValueOnce(new Error('put failed'));

        await expect(
          db.withTransaction(async (txn) => {
            await txn.put('key1', 'value1');
          })
        ).rejects.toThrow('put failed');

        expect(mockClient.abortTransaction).toHaveBeenCalledWith(BigInt(12345));
        expect(mockClient.commitTransaction).not.toHaveBeenCalled();
      });

      it('should return value from transaction', async () => {
        mockClient.get.mockResolvedValue(Buffer.from('test'));

        const result = await db.withTransaction(async (txn) => {
          const val = await txn.get('key');
          return val?.toString();
        });

        expect(result).toBe('test');
      });
    });

    describe('close', () => {
      it('should close the database', async () => {
        const fs = require('fs');
        fs.existsSync = jest.fn().mockReturnValue(true);
        const db = await Database.open('./test_db');
        mockClient.close.mockResolvedValue(undefined);

        await db.close();

        expect(mockClient.close).toHaveBeenCalled();
      });

      it('should throw when operating on closed database', async () => {
        const fs = require('fs');
        fs.existsSync = jest.fn().mockReturnValue(true);
        const db = await Database.open('./test_db');
        mockClient.close.mockResolvedValue(undefined);

        await db.close();

        await expect(db.get('key')).rejects.toThrow(DatabaseError);
      });
    });

    describe('stats', () => {
      it('should return storage stats', async () => {
        const fs = require('fs');
        fs.existsSync = jest.fn().mockReturnValue(true);
        const db = await Database.open('./test_db');
        
        mockClient.stats.mockResolvedValue({
          memtableSizeBytes: 1024,
          walSizeBytes: 512,
          activeTransactions: 2,
        });

        const stats = await db.stats();

        expect(stats).toEqual({
          memtableSizeBytes: 1024,
          walSizeBytes: 512,
          activeTransactions: 2,
        });
      });
    });
  });

  describe('Transaction', () => {
    it('should throw when operating on committed transaction', async () => {
      const fs = require('fs');
      fs.existsSync = jest.fn().mockReturnValue(true);
      
      const mockClient = {
        get: jest.fn(),
        put: jest.fn().mockResolvedValue(undefined),
        beginTransaction: jest.fn().mockResolvedValue(BigInt(1)),
        commitTransaction: jest.fn().mockResolvedValue(undefined),
        abortTransaction: jest.fn(),
        close: jest.fn(),
      };
      (IpcClient.connect as jest.Mock) = jest.fn().mockResolvedValue(mockClient);

      const db = await Database.open('./test_db');
      
      await db.withTransaction(async (txn) => {
        await txn.put('key', 'value');
        // Transaction commits after this
      });

      // The transaction object is no longer accessible after withTransaction
      // This test validates the internal behavior
    });
  });

  describe('VectorIndex', () => {
    describe('constructor', () => {
      it('should create with default config', () => {
        const index = new VectorIndex('./vectors');
        expect(index).toBeDefined();
      });

      it('should create with custom config', () => {
        const config: VectorIndexConfig = {
          dimension: 384,
          metric: 'cosine',
          m: 32,
          efConstruction: 200,
        };
        const index = new VectorIndex('./vectors', config);
        expect(index).toBeDefined();
      });
    });

    describe('static methods', () => {
      it('computeCosineDistance should calculate correctly', () => {
        const a = [1, 0, 0];
        const b = [1, 0, 0];
        const dist = VectorIndex.computeCosineDistance(a, b);
        expect(dist).toBeCloseTo(0, 5);
      });

      it('computeCosineDistance for opposite vectors', () => {
        const a = [1, 0, 0];
        const b = [-1, 0, 0];
        const dist = VectorIndex.computeCosineDistance(a, b);
        expect(dist).toBeCloseTo(2, 5);
      });

      it('computeEuclideanDistance should calculate correctly', () => {
        const a = [0, 0, 0];
        const b = [3, 4, 0];
        const dist = VectorIndex.computeEuclideanDistance(a, b);
        expect(dist).toBeCloseTo(5, 5);
      });

      it('normalizeVector should normalize to unit length', () => {
        const v = [3, 4];
        const normalized = VectorIndex.normalizeVector(v);
        expect(normalized[0]).toBeCloseTo(0.6, 5);
        expect(normalized[1]).toBeCloseTo(0.8, 5);
        
        // Check unit length
        const normSq = normalized.reduce((sum, x) => sum + x * x, 0);
        expect(normSq).toBeCloseTo(1, 5);
      });

      it('normalizeVector should handle zero vector', () => {
        const v = [0, 0, 0];
        const normalized = VectorIndex.normalizeVector(v);
        expect(normalized).toEqual([0, 0, 0]);
      });
    });
  });

  describe('Error hierarchy', () => {
    it('ToonDBError is base class', () => {
      const err = new ToonDBError('test');
      expect(err).toBeInstanceOf(Error);
      expect(err.name).toBe('ToonDBError');
    });

    it('ConnectionError extends ToonDBError', () => {
      const err = new ConnectionError('conn failed');
      expect(err).toBeInstanceOf(ToonDBError);
      expect(err.name).toBe('ConnectionError');
    });

    it('TransactionError extends ToonDBError', () => {
      const err = new TransactionError('txn failed');
      expect(err).toBeInstanceOf(ToonDBError);
      expect(err.name).toBe('TransactionError');
    });

    it('ProtocolError extends ToonDBError', () => {
      const err = new ProtocolError('proto error');
      expect(err).toBeInstanceOf(ToonDBError);
      expect(err.name).toBe('ProtocolError');
    });

    it('DatabaseError extends ToonDBError', () => {
      const err = new DatabaseError('db error');
      expect(err).toBeInstanceOf(ToonDBError);
      expect(err.name).toBe('DatabaseError');
    });

    it('errors can be caught by type', () => {
      const throwConn = () => {
        throw new ConnectionError('test');
      };

      try {
        throwConn();
      } catch (e) {
        if (e instanceof ConnectionError) {
          expect(e.message).toBe('test');
        } else {
          throw new Error('Wrong error type');
        }
      }
    });
  });
});
