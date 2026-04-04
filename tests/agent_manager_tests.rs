use gypsy::agent_manager::AgentManager;
use mentalist::DeepAgentState;
use gypsy::config::AppConfig;
use mem_core::{EmbeddingProvider, TokenCounter, Response, ResponseChunk, ModelProvider, Request};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::mpsc;
use futures_util::stream::{self, BoxStream};
use std::path::PathBuf;

// --- Mock Providers ---

struct MockProvider;

#[async_trait]
impl ModelProvider for MockProvider {
    async fn complete(&self, _req: Request) -> anyhow::Result<Response> {
        Ok(Response { content: "Mocked Response".into(), tool_calls: vec![] })
    }
    async fn stream_complete(&self, _req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        let chunk = ResponseChunk {
            content_delta: Some("Chunk1".into()),
            tool_call_delta: None,
            usage: None,
            is_final: true,
        };
        Ok(Box::pin(stream::iter(vec![Ok(chunk)])))
    }
}

struct MockEmbed;
#[async_trait]
impl EmbeddingProvider for MockEmbed {
    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> { Ok(vec![0.1; 384]) }
}

struct MockCounter;
impl TokenCounter for MockCounter {
    fn count_tokens(&self, _text: &str) -> usize { 10 }
}

#[tokio::test]
async fn test_agent_manager_init() {
    let (tx, _rx) = mpsc::unbounded_channel();
    let config = AppConfig::from_env();
    
    let manager = AgentManager::new(
        tx,
        Box::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        config
    );
    
    assert!(manager.is_ok());
}

#[tokio::test]
async fn test_agent_manager_session_persistence() {
    let (tx, _rx) = mpsc::unbounded_channel();
    let session_id = "test_persistence_001".to_string();
    
    // 1. Create agent and ensure session dir exists
    let storage_dir = PathBuf::from(".agent/sessions");
    std::fs::create_dir_all(&storage_dir).unwrap();
    
    let mut config = AppConfig::from_env();
    config.session_id = session_id.clone();
    
    let manager = AgentManager::new(
        tx.clone(),
        Box::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        config.clone()
    ).unwrap();
    
    // 2. Modify state (add context item)
    let mut current_ctx = (*manager.agent.state.context).clone();
    current_ctx.items.push(mem_core::MemoryItem {
        role: mem_core::MemoryRole::User,
        content: "Save me".into(),
        timestamp: 12345,
        metadata: serde_json::json!({}),
    });
    // Arc is immutable, we must replace it
    // Note: In real scenarios, deep_agent handles this via its own logic, 
    // but here we are manually touching the state.
    unsafe {
        let ptr = &manager.agent.state as *const DeepAgentState as *mut DeepAgentState;
        (*ptr).context = Arc::new(current_ctx);
    }
    
    // 3. Force save
    let state_file = storage_dir.join(format!("session_{}.json", session_id));
    let data = serde_json::to_string(&manager.agent.state).unwrap();
    std::fs::write(&state_file, data).unwrap();
    
    // 4. Re-init with same ID and verify
    let manager2 = AgentManager::new(
        tx,
        Box::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        config
    ).unwrap();
    
    assert_eq!(manager2.agent.state.context.items.len(), 1);
    assert_eq!(manager2.agent.state.context.items[0].content, "Save me");
    
    // Cleanup
    std::fs::remove_file(state_file).ok();
}
