use async_trait::async_trait;
use gypsy::agent_manager::{AgentEvent, AgentManager};
use gypsy::config::AppConfig;
use gypsy::prefs::PrefsManager;
use mem_core::{
    EmbeddingProvider, MemoryItem, MemoryRole, ModelProvider, Request, Response, TokenCounter,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

// --- Mock Providers (Copied from agent_manager_tests.rs) ---

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
    ) -> anyhow::Result<
        futures_util::stream::BoxStream<'static, anyhow::Result<mem_core::ResponseChunk>>,
    > {
        let chunk = mem_core::ResponseChunk {
            content_delta: Some("Chunk1".into()),
            tool_call_delta: None,
            usage: None,
            is_final: false,
        };
        Ok(Box::pin(futures_util::stream::iter(vec![Ok(chunk)])))
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

fn make_prefs() -> Arc<tokio::sync::Mutex<PrefsManager>> {
    Arc::new(tokio::sync::Mutex::new(PrefsManager::new(&PathBuf::from(
        "/tmp",
    ))))
}

#[tokio::test]
async fn test_skill_discovery_valid() {
    let (tx, _rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();

    // Create a temporary skills directory
    let temp_dir = std::env::current_dir().unwrap().join("test_skills_valid");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let skill_dir = temp_dir.join("valid-skill");
    std::fs::create_dir_all(&skill_dir).unwrap();

    let skill_md = r#"---
name: valid-skill
description: A valid skill test.
---
# Skill Content"#;
    std::fs::write(skill_dir.join("SKILL.md"), skill_md).unwrap();

    config.skills_path = temp_dir.to_string_lossy().to_string();
    config.storage_path = temp_dir.join("storage").to_string_lossy().to_string();
    config.sessions_path = temp_dir.join("sessions").to_string_lossy().to_string();
    config.mcp_root_path = std::env::current_dir()
        .unwrap()
        .join(".agent/mcp")
        .to_string_lossy()
        .to_string();

    let manager = AgentManager::new(
        tx,
        Arc::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        make_prefs(),
        config,
    )
    .await
    .unwrap();

    assert_eq!(manager.discovered_skills.len(), 1);
    assert_eq!(manager.discovered_skills[0].name, "valid-skill");
    assert_eq!(
        manager.discovered_skills[0].description,
        "A valid skill test."
    );

    // Cleanup
    //let _ = std::fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn test_skills_command_json() {
    let (tx, mut rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();

    let temp_dir = std::env::current_dir().unwrap().join("test_skills_cmd");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let skill_dir = temp_dir.join("test-skill-123");
    std::fs::create_dir_all(&skill_dir).unwrap();

    let skill_md = r#"---
name: ignored-name
description: Test skill command
---"#;
    std::fs::write(skill_dir.join("SKILL.md"), skill_md).unwrap();

    config.skills_path = temp_dir.to_string_lossy().to_string();
    config.storage_path = temp_dir.join("storage").to_string_lossy().to_string();
    config.sessions_path = temp_dir.join("sessions").to_string_lossy().to_string();
    config.mcp_root_path = std::env::current_dir()
        .unwrap()
        .join(".agent/mcp")
        .to_string_lossy()
        .to_string();

    let mut manager = AgentManager::new(
        tx,
        Arc::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        make_prefs(),
        config,
    )
    .await
    .unwrap();

    manager.handle_command("/skills").await.unwrap();

    let mut found = false;
    while let Some(event) = rx.recv().await {
        if let AgentEvent::TextChunk(msg) = event {
            if msg.contains("### Discovered Skills (agentskills.io)") &&
               msg.contains("test-skill-123") &&
               msg.contains("Test skill command") {
                found = true;
                break;
            }
        }
    }
    assert!(found, "Expected TextChunk for /skills with correct content");

    // Cleanup
    //let _ = std::fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn test_summarize_command() {
    let (tx, mut rx) = mpsc::channel(1000);
    let mut config = AppConfig::from_env();

    let temp_dir = std::env::current_dir().unwrap().join("test_summarize");
    let _ = std::fs::remove_dir_all(&temp_dir);

    config.storage_path = temp_dir.join("storage").to_string_lossy().to_string();
    config.sessions_path = temp_dir.join("sessions").to_string_lossy().to_string();
    config.mcp_root_path = std::env::current_dir()
        .unwrap()
        .join(".agent/mcp")
        .to_string_lossy()
        .to_string();
    config.session_id = "test_summary_01".into();

    let mut manager = AgentManager::new(
        tx,
        Arc::new(MockProvider),
        Arc::new(MockEmbed),
        Arc::new(MockCounter),
        make_prefs(),
        config,
    )
    .await
    .unwrap();

    // Setup a session with some items
    let mut state = manager
        .session_manager
        .load_session("test_summary_01")
        .unwrap();
    state.context = Arc::new(mem_core::Context {
        items: (0..10)
            .map(|i| MemoryItem {
                role: MemoryRole::User,
                content: format!("Message {}", i),
                timestamp: i as u64,
                metadata: serde_json::json!({}),
            })
            .collect(),
    });
    manager.session_manager.save_session(&state).unwrap();

    // Since we use MockProvider, the compactor won't actually do much if it requires LLM.
    // However, we verify the command executes and doesn't crash.
    let found = Arc::new(tokio::sync::Mutex::new(false));
    let found_clone = Arc::clone(&found);
    let handle = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if let AgentEvent::TextChunk(msg) = event {
                if msg.contains("Context optimized") {
                    *found_clone.lock().await = true;
                }
            }
        }
    });

    manager.handle_command("/summarize").await.unwrap();
    drop(manager);
    let _ = handle.await;

    assert!(*found.lock().await, "Context optimized chunk not received");

    // Cleanup
    //let _ = std::fs::remove_dir_all(&temp_dir);
}
