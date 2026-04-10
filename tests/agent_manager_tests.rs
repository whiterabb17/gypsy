use async_trait::async_trait;
use futures_util::stream::{self, BoxStream};
use gypsy::agent_manager::{AgentEvent, AgentManager};
use gypsy::config::AppConfig;
use gypsy::prefs::PrefsManager;
use mem_core::{
    EmbeddingProvider, MemoryItem, MemoryRole, ModelProvider, Request, Response, ResponseChunk,
    TokenCounter,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

// --- Mock Providers ---

struct MockProvider;

#[async_trait]
impl mem_core::LlmClient for MockProvider {
    async fn completion(&self, _prompt: &str) -> anyhow::Result<String> {
        Ok("Mock Completion".into())
    }
}

#[async_trait]
impl ModelProvider for MockProvider {
    async fn complete(&self, _req: Request) -> anyhow::Result<Response> {
        Ok(Response {
            content: "Mocked Response".into(),
            tool_calls: vec![],
            usage: None,
        })
    }
    async fn stream_complete(
        &self,
        _req: Request,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        let chunk = ResponseChunk {
            content_delta: Some("Chunk1".into()),
            tool_call_delta: None,
            usage: None,
            is_final: false,
        };
        let final_chunk = ResponseChunk {
            content_delta: None,
            tool_call_delta: None,
            usage: None,
            is_final: true,
        };
        Ok(Box::pin(stream::iter(vec![Ok(chunk), Ok(final_chunk)])))
    }
}

struct MockEmbed;
#[async_trait]
impl EmbeddingProvider for MockEmbed {
    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.1; 768])
    }
}

struct MockCounter;
impl TokenCounter for MockCounter {
    fn count_tokens(&self, _text: &str) -> usize {
        10
    }
}


#[tokio::test]
async fn test_agent_manager_init() {
    let (tx, _rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();
    config.storage_path = "/tmp/gypsy_test_storage".into();
    config.mcp_root_path = std::env::current_dir()
        .unwrap()
        .join(".agent/mcp")
        .to_string_lossy()
        .to_string();
    config.max_steps = 10;

    let prefs = Arc::new(tokio::sync::Mutex::new(PrefsManager::new(&PathBuf::from("/tmp"))));

    let manager = AgentManager::new(
        tx,
        Arc::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        prefs,
        config,
        None,
        None,
    )

    .await;

    assert!(manager.is_ok());
}

#[tokio::test]
async fn test_tool_registry_ready() {
    let (tx, _rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();
    config.storage_path = "/tmp/gypsy_test_registry".into();
    config.mcp_root_path = "/tmp/gypsy_test_registry/mcp".into();
    config.max_steps = 10;

    let prefs = Arc::new(tokio::sync::Mutex::new(PrefsManager::new(&PathBuf::from("/tmp"))));

    let manager = AgentManager::new(
        tx,
        Arc::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        prefs,
        config,
        None,
        None,
    )

    .await
    .unwrap();

    // Verify we can add a manual tool and it appears in the registry
    use mentalist::tools::mcp_adapter::{McpServer, McpTool};
    manager
        .multi_executor
        .add_tool(Arc::new(McpTool {
            server: Arc::new(McpServer::new("mock".into(), "echo".into(), vec![])),
            source: "manual".into(),
            name: "registry_test".into(),
            description: "test".into(),
            parameters: serde_json::json!({}),
        }))
        .await;

    let tools = manager.multi_executor.registry.list_tools().await;
    assert!(tools.iter().any(|t| t.name == "registry_test"));
}


#[tokio::test]
async fn test_command_handler() {
    let (tx, mut rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();
    config.storage_path = "/tmp/gypsy_test_cmd".into();
    config.mcp_root_path = std::env::current_dir()
        .unwrap()
        .join(".agent/mcp")
        .to_string_lossy()
        .to_string();
    config.max_steps = 10;

    let prefs = Arc::new(tokio::sync::Mutex::new(PrefsManager::new(&PathBuf::from("/tmp"))));

    let mut manager = AgentManager::new(
        tx,
        Arc::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        prefs,
        config,
        None,
        None,
    )

    .await
    .unwrap();

    // Add a manual tool to verification registry directly
    use mentalist::tools::mcp_adapter::{McpServer, McpTool};
    manager
        .multi_executor
        .add_tool(Arc::new(McpTool {
            server: Arc::new(McpServer::new("mock".into(), "echo".into(), vec![])),
            source: "mock".into(),
            name: "test_tool".into(),
            description: "test".into(),
            parameters: serde_json::json!({}),
        }))
        .await;

    // Record events concurrently
    let found = Arc::new(tokio::sync::Mutex::new(false));
    let found_clone = Arc::clone(&found);
    let handle = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if let AgentEvent::TextChunk(msg) = event {
                if msg.contains("### Available Tools") && msg.contains("test_tool") {
                    *found_clone.lock().await = true;
                }
            }
        }
    });

    // Test /tools
    manager.run_step("/tools".to_string()).await.unwrap();
    drop(manager);
    let _ = handle.await;

    assert!(*found.lock().await, "Expected TextChunk event for /tools with test_tool");
}

#[tokio::test]
async fn test_agent_manager_session_persistence() {
    let (tx, _rx) = mpsc::channel(1000);
    let session_id = "test_persistence_002".to_string();
    let storage_dir = std::env::current_dir().unwrap().join("test_sessions");
    std::fs::create_dir_all(&storage_dir).unwrap();

    let mut config = AppConfig::from_env();
    config.sessions_path = storage_dir.to_string_lossy().to_string();
    config.session_id = session_id.clone();
    config.max_steps = 10;
    config.mcp_root_path = storage_dir.join("mcp").to_string_lossy().to_string();

    let prefs = Arc::new(tokio::sync::Mutex::new(PrefsManager::new(&PathBuf::from("/tmp"))));

    let manager = AgentManager::new(
        tx.clone(),
        Arc::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        prefs,
        config.clone(),
        None,
        None,
    )

    .await
    .unwrap();

    // In v0.3.5, we verify persistence via the SessionManager directly
    let state = mentalist::AgentState {
        session_id: session_id.clone(),
        context: Arc::new(mem_core::Context {
            items: vec![MemoryItem {
                role: MemoryRole::User,
                content: "Persistence test content".into(),
                timestamp: 12345,
                metadata: serde_json::json!({}),
            }],
        }),
        ..Default::default()
    };

    manager.session_manager.save_session(&state).unwrap();

    // Re-init with same ID and verify
    let prefs2 = Arc::new(tokio::sync::Mutex::new(PrefsManager::new(&PathBuf::from("/tmp"))));

    let manager2 = AgentManager::new(
        tx,
        Arc::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        prefs2,
        config,
        None,
        None,
    )

    .await
    .unwrap();

    let loaded_state = manager2.session_manager.load_session(&session_id).unwrap();
    assert_eq!(loaded_state.context.items.len(), 2);
    assert_eq!(
        loaded_state.context.items[1].content,
        "Persistence test content"
    );

    // Cleanup
    //std::fs::remove_dir_all(storage_dir).ok();
}

#[tokio::test]
async fn test_vector_memory_recall() {
    let (tx, _rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();
    let storage_dir = std::env::current_dir().unwrap().join("test_vector");
    config.storage_path = storage_dir.to_string_lossy().to_string();
    config.mcp_root_path = storage_dir.join("mcp").to_string_lossy().to_string();
    config.max_steps = 10;

    let prefs = Arc::new(tokio::sync::Mutex::new(PrefsManager::new(&PathBuf::from("/tmp"))));

    let manager = AgentManager::new(
        tx,
        Arc::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        prefs,
        config,
        None,
        None,
    )

    .await
    .unwrap();

    // Store a "memory" via the mentalist memory interface

    // For this test, we verify that we can at least invoke the recall method
    let query = mentalist::memory::MemoryQuery {
        text: "password".into(),
        limit: 1,
    };

    let results = manager.runtime.memory.recall(query).await;
    assert!(results.is_ok());
}

#[tokio::test]
async fn test_consume_command() {
    let (tx, mut rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();
    let temp_dir = tempfile::tempdir().unwrap();
    config.storage_path = temp_dir.path().join("storage").to_string_lossy().to_string();
    config.sessions_path = temp_dir.path().join("sessions").to_string_lossy().to_string();
    config.mcp_root_path = std::env::current_dir().unwrap().join(".agent/mcp").to_string_lossy().to_string();
    config.session_id = "test_consume_01".into();
    config.max_steps = 10;

    // Create a dummy file to consume
    let dummy_file = temp_dir.path().join("dummy.rs");
    std::fs::write(&dummy_file, "fn test() { println!(\"consumed\"); }").unwrap();

    let prefs = Arc::new(tokio::sync::Mutex::new(PrefsManager::new(&PathBuf::from("/tmp"))));

    let mut manager = AgentManager::new(
        tx,
        Arc::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        prefs,
        config,
        None,
        None,
    )

    .await
    .unwrap();

    // Change current dir to temp_dir to simulate consumption
    let original_dir = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp_dir.path()).unwrap();

    // 1. Check Events concurrently
    let chunk_received = Arc::new(tokio::sync::Mutex::new(false));
    let chunk_received_clone = Arc::clone(&chunk_received);
    let handle = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if let AgentEvent::TextChunk(c) = event {
                if c.contains("Studied") && c.contains("Knowledge base updated") {
                    *chunk_received_clone.lock().await = true;
                }
            }
        }
    });

    manager.handle_command("/consume").await.unwrap();

    // Restore dir
    std::env::set_current_dir(original_dir).unwrap();

    // 2. Check Session Context for Knowledge Update
    let loaded = manager.session_manager.load_session("test_consume_01").unwrap();
    let has_update = loaded.context.items.iter().any(|item| {
        item.metadata.get("type").and_then(|v| v.as_str()) == Some("knowledge_update")
            && item.content.contains("dummy.rs")
    });

    drop(manager);
    let _ = handle.await;

    assert!(*chunk_received.lock().await, "Consumption status chunk not received with 'Studied' and 'Knowledge base updated'");
    assert!(has_update, "Context does not contain knowledge update with file list");
}
