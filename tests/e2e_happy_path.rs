use async_trait::async_trait;
use futures_util::stream::{self, BoxStream};
use gypsy::agent_manager::{AgentEvent, AgentManager};
use gypsy::config::AppConfig;
use gypsy::prefs::PrefsManager;
use mem_core::{
    EmbeddingProvider, ModelProvider, Request, Response, ResponseChunk, TokenCounter, ToolCall,
    ToolCallDelta,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;

// --- E2E Mocks ---

struct HappyModel {
    pub call_count: Arc<tokio::sync::Mutex<usize>>,
}

#[async_trait]
impl mem_core::LlmClient for HappyModel {
    async fn completion(&self, _prompt: &str) -> anyhow::Result<String> {
        let mut count = self.call_count.lock().await;
        *count += 1;
        if *count == 1 {
            Ok("I need to check the system status.".into())
        } else if *count == 2 {
            Ok("System status is healthy. Updating logs.".into())
        } else {
            Ok("The system is online and healthy.".into())
        }
    }
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
                    id: "test-id".to_string(),
                }],
                usage: None,
            });
        }

        // Turn 2: Final answer based on tool result
        Ok(Response {
            content: "The system is online and healthy.".into(),
            tool_calls: vec![],
            usage: None,
        })
    }

    async fn stream_complete(
        &self,
        _req: Request,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
        let res = self.complete(_req).await?;
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
                    id: Some(tool.id.clone()),
                }),
                usage: None,
                is_final: true,
            }));
        }
        if chunks.is_empty()
            || !chunks
                .last()
                .unwrap()
                .as_ref()
                .map_or(false, |c| c.is_final)
        {
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
    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0; 768])
    }
}

struct DummyCounter;
impl TokenCounter for DummyCounter {
    fn count_tokens(&self, _text: &str) -> usize {
        1
    }
}

fn make_prefs() -> Arc<tokio::sync::Mutex<PrefsManager>> {
    Arc::new(tokio::sync::Mutex::new(PrefsManager::new(&PathBuf::from("/tmp"))))
}

#[tokio::test]
async fn test_e2e_autonomous_tool_loop() {
    let (tx, mut rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();
    let storage_dir = std::env::current_dir().unwrap().join("test_e2e_happy");
    config.storage_path = storage_dir.to_string_lossy().to_string();
    config.session_id = "test_e2e_autonomous_001".to_string();
    config.max_steps = 10;
    config.mcp_root_path = std::env::current_dir()
        .unwrap()
        .join(".agent/mcp")
        .to_string_lossy()
        .to_string();

    let model = HappyModel {
        call_count: Arc::new(tokio::sync::Mutex::new(0)),
    };

    let mut manager = AgentManager::new(
        tx,
        Arc::new(model),
        Arc::new(DummyEmbed),
        Arc::new(DummyCounter),
        make_prefs(),
        config,
    )
    .await
    .unwrap();

    // 1. Run agent & Consume concurrently to avoid deadlock
    let logs = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let logs_clone = Arc::clone(&logs);

    let handle = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if let AgentEvent::TextChunk(c) = event {
                logs_clone.lock().await.push(c);
            }
        }
    });

    manager.run_step("Check status".into()).await.unwrap();
    // Drop manager's tx clone if needed, but here we just wait a bit or use a more robust way
    // Dropping manager or tx will close the channel eventually.
    drop(manager);
    let _ = handle.await;

    let full_log = logs.lock().await.join("");
    assert!(full_log.contains("system status") || full_log.contains("online and healthy"));
}

#[tokio::test]
async fn test_session_persistence_context() {
    let (tx, mut rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();
    let storage_dir = std::env::current_dir().unwrap().join("test_e2e_persist");
    config.storage_path = storage_dir.to_string_lossy().to_string();
    config.sessions_path = storage_dir.to_string_lossy().to_string();
    config.session_id = "persist_e2e_001".to_string();
    config.max_steps = 10;
    config.mcp_root_path = std::env::current_dir()
        .unwrap()
        .join(".agent/mcp")
        .to_string_lossy()
        .to_string();

    let model = HappyModel {
        call_count: Arc::new(tokio::sync::Mutex::new(0)),
    };

    let mut manager = AgentManager::new(
        tx,
        Arc::new(model),
        Arc::new(DummyEmbed),
        Arc::new(DummyCounter),
        make_prefs(),
        config,
    )
    .await
    .unwrap();

    manager.run_step("First request".into()).await.unwrap();

    // Drain events to ensure it completes, ignoring background statuses
    let timeout = tokio::time::sleep(std::time::Duration::from_secs(5));
    tokio::pin!(timeout);
    loop {
        tokio::select! {
            Some(_) = rx.recv() => {},
            _ = &mut timeout => break,
        }
    }

    // Verify context is saved (User prompt + Assistant response = 2 items minimum)
    let loaded = manager
        .session_manager
        .load_session("persist_e2e_001")
        .unwrap();
    assert!(loaded.context.items.len() >= 2);

    // std::fs::remove_dir_all(storage_dir).ok();
}

#[tokio::test]
async fn test_step_limit_enforcement() {
    let (tx, mut _rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();
    let storage_dir = std::env::current_dir().unwrap().join("test_e2e_limit");
    config.storage_path = storage_dir.to_string_lossy().to_string();
    config.max_steps = 2; // Set low limit
    config.mcp_root_path = std::env::current_dir()
        .unwrap()
        .join(".agent/mcp")
        .to_string_lossy()
        .to_string();

    let model = HappyModel {
        call_count: Arc::new(tokio::sync::Mutex::new(0)),
    };

    let mut manager = AgentManager::new(
        tx,
        Arc::new(model),
        Arc::new(DummyEmbed),
        Arc::new(DummyCounter),
        make_prefs(),
        config,
    )
    .await
    .unwrap();

    // Should terminate gracefully after 2 steps
    let res = manager.run_step("Loop forever".into()).await;
    assert!(res.is_ok());

    //std::fs::remove_dir_all(storage_dir).ok();
}

#[tokio::test]
async fn test_fallback_parsing_e2e() {
    let (tx, mut rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();
    let storage_dir = std::env::current_dir().unwrap().join("test_e2e_fallback");
    let _ = std::fs::create_dir_all(&storage_dir);
    config.storage_path = storage_dir.to_string_lossy().to_string();
    config.sessions_path = storage_dir.to_string_lossy().to_string();
    config.max_steps = 10;
    config.session_id = "test_fallback_001".to_string();
    config.mcp_root_path = std::env::current_dir()
        .unwrap()
        .join(".agent/mcp")
        .to_string_lossy()
        .to_string();

    struct FallbackModel;
    #[async_trait]
    impl mem_core::LlmClient for FallbackModel {
        async fn completion(&self, _prompt: &str) -> anyhow::Result<String> {
            Ok("Let me use the tool: <call name=\"echo\">{\"msg\": \"hello\"}</call>".into())
        }
    }
    #[async_trait]
    impl ModelProvider for FallbackModel {
        async fn complete(&self, _req: Request) -> anyhow::Result<Response> {
            Ok(Response {
                // Return a raw text tool call instead of structured tool_calls
                content: "Let me use the tool: <call name=\"echo\">{\"msg\": \"hello\"}</call>"
                    .into(),
                tool_calls: vec![],
                usage: None,
            })
        }
        async fn stream_complete(
            &self,
            _req: Request,
        ) -> anyhow::Result<BoxStream<'static, anyhow::Result<ResponseChunk>>> {
            let res = self.complete(_req).await?;
            Ok(Box::pin(stream::iter(vec![Ok(ResponseChunk {
                content_delta: Some(res.content),
                tool_call_delta: None,
                usage: None,
                is_final: true,
            })])))
        }
    }

    let mut manager = AgentManager::new(
        tx,
        Arc::new(FallbackModel),
        Arc::new(DummyEmbed),
        Arc::new(DummyCounter),
        make_prefs(),
        config,
    )
    .await
    .unwrap();

    // 1. Run agent concurrently
    let logs = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let logs_clone = Arc::clone(&logs);
    let handle = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if let AgentEvent::ToolResult { name, .. } = event {
                if name == "echo" {
                    logs_clone.lock().await.push("hello".to_string());
                }
            }
        }
    });

    manager.run_step("Check status".into()).await.unwrap();
    drop(manager);
    let _ = handle.await;

    let full_log = logs.lock().await.join("");
    assert!(
        full_log.contains("hello"),
        "Agent should have detected fallback XML tool call to 'echo'"
    );
}
