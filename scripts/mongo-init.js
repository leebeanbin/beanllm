// MongoDB Initialization Script for beanllm
// This script runs on first container startup

// Switch to beanllm database
db = db.getSiblingDB('beanllm');

// Create collections with validation schemas

// ===========================================
// Chat Sessions Collection
// ===========================================
db.createCollection('chat_sessions', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['session_id', 'created_at'],
      properties: {
        session_id: {
          bsonType: 'string',
          description: 'Unique session identifier'
        },
        title: {
          bsonType: 'string',
          description: 'Session title (auto-generated or user-defined)'
        },
        feature_mode: {
          bsonType: 'string',
          enum: ['chat', 'rag', 'multi-agent', 'knowledge-graph', 'audio', 'ocr', 'google', 'web-search', 'evaluation', 'agentic'],
          description: 'Feature mode used in this session'
        },
        model: {
          bsonType: 'string',
          description: 'Primary model used'
        },
        messages: {
          bsonType: 'array',
          items: {
            bsonType: 'object',
            required: ['role', 'content', 'timestamp'],
            properties: {
              role: {
                bsonType: 'string',
                enum: ['system', 'user', 'assistant', 'tool']
              },
              content: {
                bsonType: 'string'
              },
              timestamp: {
                bsonType: 'date'
              },
              model: {
                bsonType: 'string'
              },
              usage: {
                bsonType: 'object'
              },
              tool_calls: {
                bsonType: 'array'
              },
              metadata: {
                bsonType: 'object'
              }
            }
          }
        },
        feature_options: {
          bsonType: 'object',
          description: 'Feature-specific options'
        },
        created_at: {
          bsonType: 'date'
        },
        updated_at: {
          bsonType: 'date'
        },
        total_tokens: {
          bsonType: 'int'
        },
        message_count: {
          bsonType: 'int'
        }
      }
    }
  }
});

// Create indexes for chat_sessions
db.chat_sessions.createIndex({ session_id: 1 }, { unique: true });
db.chat_sessions.createIndex({ created_at: -1 });
db.chat_sessions.createIndex({ updated_at: -1 });
db.chat_sessions.createIndex({ feature_mode: 1 });
db.chat_sessions.createIndex({ model: 1 });

// ===========================================
// API Keys Collection (Encrypted Storage)
// ===========================================
db.createCollection('api_keys', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['provider', 'key_encrypted', 'created_at'],
      properties: {
        provider: {
          bsonType: 'string',
          enum: ['openai', 'anthropic', 'google', 'gemini', 'ollama', 'deepseek', 'perplexity', 'tavily', 'serpapi', 'pinecone', 'qdrant', 'weaviate', 'neo4j', 'google_oauth'],
          description: 'Provider name'
        },
        key_encrypted: {
          bsonType: 'string',
          description: 'Encrypted API key'
        },
        key_hint: {
          bsonType: 'string',
          description: 'Last 4 characters for identification'
        },
        is_valid: {
          bsonType: 'bool',
          description: 'Whether the key has been validated'
        },
        last_validated: {
          bsonType: 'date'
        },
        created_at: {
          bsonType: 'date'
        },
        updated_at: {
          bsonType: 'date'
        },
        metadata: {
          bsonType: 'object',
          description: 'Additional metadata (e.g., organization, project)'
        }
      }
    }
  }
});

// Create indexes for api_keys
db.api_keys.createIndex({ provider: 1 }, { unique: true });
db.api_keys.createIndex({ is_valid: 1 });

// ===========================================
// Google OAuth Tokens Collection
// ===========================================
db.createCollection('google_oauth_tokens', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['user_id', 'access_token_encrypted', 'created_at'],
      properties: {
        user_id: {
          bsonType: 'string',
          description: 'User identifier (default: "default" for single-user mode)'
        },
        access_token_encrypted: {
          bsonType: 'string'
        },
        refresh_token_encrypted: {
          bsonType: 'string'
        },
        token_type: {
          bsonType: 'string'
        },
        expires_at: {
          bsonType: 'date'
        },
        scopes: {
          bsonType: 'array',
          items: {
            bsonType: 'string'
          }
        },
        created_at: {
          bsonType: 'date'
        },
        updated_at: {
          bsonType: 'date'
        }
      }
    }
  }
});

// Create indexes for google_oauth_tokens
db.google_oauth_tokens.createIndex({ user_id: 1 }, { unique: true });
db.google_oauth_tokens.createIndex({ expires_at: 1 });

// ===========================================
// Request Logs Collection (for monitoring)
// ===========================================
db.createCollection('request_logs', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['request_id', 'timestamp'],
      properties: {
        request_id: {
          bsonType: 'string'
        },
        session_id: {
          bsonType: 'string'
        },
        endpoint: {
          bsonType: 'string'
        },
        method: {
          bsonType: 'string'
        },
        status_code: {
          bsonType: 'int'
        },
        duration_ms: {
          bsonType: 'double'
        },
        model: {
          bsonType: 'string'
        },
        provider: {
          bsonType: 'string'
        },
        input_tokens: {
          bsonType: 'int'
        },
        output_tokens: {
          bsonType: 'int'
        },
        total_tokens: {
          bsonType: 'int'
        },
        error: {
          bsonType: 'string'
        },
        timestamp: {
          bsonType: 'date'
        }
      }
    }
  }
});

// Create indexes for request_logs
db.request_logs.createIndex({ request_id: 1 }, { unique: true });
db.request_logs.createIndex({ session_id: 1 });
db.request_logs.createIndex({ timestamp: -1 });
db.request_logs.createIndex({ endpoint: 1, timestamp: -1 });
db.request_logs.createIndex({ model: 1 });
db.request_logs.createIndex({ status_code: 1 });

// TTL index to auto-delete old logs after 30 days
db.request_logs.createIndex(
  { timestamp: 1 },
  { expireAfterSeconds: 2592000 }  // 30 days
);

// ===========================================
// RAG Documents Collection (optional, for local storage)
// ===========================================
db.createCollection('rag_documents', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['doc_id', 'content', 'created_at'],
      properties: {
        doc_id: {
          bsonType: 'string'
        },
        collection_name: {
          bsonType: 'string'
        },
        content: {
          bsonType: 'string'
        },
        metadata: {
          bsonType: 'object'
        },
        embedding: {
          bsonType: 'array'
        },
        chunk_index: {
          bsonType: 'int'
        },
        source: {
          bsonType: 'string'
        },
        created_at: {
          bsonType: 'date'
        }
      }
    }
  }
});

// Create indexes for rag_documents
db.rag_documents.createIndex({ doc_id: 1 }, { unique: true });
db.rag_documents.createIndex({ collection_name: 1 });
db.rag_documents.createIndex({ source: 1 });

print('===========================================');
print('beanllm MongoDB initialization complete!');
print('Collections created:');
print('  - chat_sessions');
print('  - api_keys');
print('  - google_oauth_tokens');
print('  - request_logs');
print('  - rag_documents');
print('===========================================');
