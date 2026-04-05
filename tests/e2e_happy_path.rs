use gypsy::agent_manager::AgentManager;
use gypsy::agent_manager::AgentEvent;
use gypsy::config::AppConfig;
use mem_core::{Request, Response, ResponseChunk, ModelProvider, EmbeddingProvider, TokenCounter, ToolCall, ToolCallDelta};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::mpsc;
use futures_util::stream::{self, BoxStream};

// --- E2E Mocks ---

struct HappyModel {
    pub call_count: Arc<tokio::sync::Mutex<usize>>,
}

#[async_trait]
impl ModelProvider for HappyModel {
    async fn complete(&self, _req: Request) -> anyhow::Result<Response> {
        let mut count = self.call_count.lock().await;
        *count += 1;

        // Turn 1: Request a tool
        if *count == 1 {
            return Ok(Response { 
                content: "I need to check the system status.".into(), 
                tool_calls: vec![ToolCall {
                    name: "echo".to_string(),
                    arguments: serde_json::json!({"msg": "system-online"}),
                }], 
            });
        }
        
        // Turn 2: Final answer based on tool result
        Ok(Response { 
            content: "The system is online and healthy.".into(), 
            tool_calls: vec![] 
        })
    }
    
    async fn stream_complete(&self, _req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        let res = self.complete(_req).await?;
        
        // Simulate streaming tool calls if present
        let mut chunks = vec![];
        if !res.content.is_empty() {
            chunks.push(Ok(ResponseChunk {
                content_delta: Some(res.content),
                tool_call_delta: None,
                usage: None,
                is_final: false,
            }));
        }
        
        for tool in res.tool_calls {
            chunks.push(Ok(ResponseChunk {
                content_delta: None,
                tool_call_delta: Some(ToolCallDelta {
                    name: Some(tool.name),
                    arguments_delta: Some(tool.arguments.to_string()),
                }),
                usage: None,
                is_final: true, // Mark this chunk as final for tool reconstruction
            }));
        }
        
        if chunks.is_empty() || !chunks.last().unwrap().as_ref().map_or(false, |c| c.is_final) {
             chunks.push(Ok(ResponseChunk {
                content_delta: None,
                tool_call_delta: None,
                usage: None,
                is_final: true,
            }));
        }

        Ok(Box::pin(stream::iter(chunks)))
    }
}

struct DummyEmbed;
#[async_trait]
impl EmbeddingProvider for DummyEmbed {
    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> { Ok(vec![0.0; 384]) }
}

struct DummyCounter;
impl TokenCounter for DummyCounter {
    fn count_tokens(&self, _text: &str) -> usize { 1 }
}

#[tokio::test]
async fn test_e2e_autonomous_tool_loop() {
    let (tx, mut rx) = mpsc::channel(100);
    let mut config = AppConfig::from_env();
    config.session_id = "test_e2e_autonomous_001".to_string();
    
    let model = HappyModel { call_count: Arc::new(tokio::sync::Mutex::new(0)) };
    
    let mut manager = AgentManager::new(
        tx,
        Arc::new(model),
        Arc::new(DummyEmbed),
        Arc::new(DummyCounter),
        config
    ).await.unwrap();
    
    // 1. Send User Prompt
    let _ = manager.run_step("Check status".into()).await.unwrap();
    
    // 2. Consume Events
    let mut logs = Vec::new();
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::TextChunk(c) => logs.push(c),
            AgentEvent::Status(s) => eprintln!("Status: {}", s),
            _ => (),
        }
    }
    
    let full_log = logs.join("");
    // Should contain both the initial thought and the final answer
    assert!(full_log.contains("system status"));
    assert!(full_log.contains("online and healthy"));
    
    // 3. Verify Memory Extraction (User, Tool Result, and Assistant response should be in context)
    // Turn 1 User + Turn 1 Assistant + Turn 2 Tool + Turn 2 Assistant = 4 items
    assert!(manager.agent.state.context.items.len() >= 3);
}
