use std::sync::Arc;
use tokio::sync::mpsc;
use anyhow::Result;
use mentalist::{Harness, DeepAgent, DeepAgentState, Request, Response, ToolCall, ModelProvider};
use mentalist::middleware::{Middleware, MindPalaceMiddleware};
use mem_core::{Context, FileStorage, EmbeddingProvider, LlmClient, TokenCounter, MemoryItem, MemoryRole};
use mem_resilience::ResilientMemoryController;
use async_trait::async_trait;
use crate::config::AppConfig;
use std::path::PathBuf;
use brain::Brain;
use mentalist::executor::ExecutionMode;

#[derive(Debug, Clone)]
pub enum AgentEvent {
    Status(String),
    TextChunk(String),
    MetricUpdate {
        tokens: usize,
        context_size: usize,
        step: String,
    },
    Error(String),
}

pub struct MonitoringMiddleware {
    tx: mpsc::UnboundedSender<AgentEvent>,
    token_counter: Arc<dyn TokenCounter>,
}

#[async_trait]
impl Middleware for MonitoringMiddleware {
    async fn before_ai_call(&self, req: &mut Request) -> Result<()> {
        let tokens: usize = req.context.items.iter().map(|i| self.token_counter.count_tokens(&i.content)).sum();
        let _ = self.tx.send(AgentEvent::MetricUpdate {
            tokens,
            context_size: req.context.items.len(),
            step: "Thinking (LLM)".to_string(),
        });
        let _ = self.tx.send(AgentEvent::Status("Thinking...".to_string()));
        Ok(())
    }

    async fn after_ai_call(&self, _res: &mut Response) -> Result<()> {
        let _ = self.tx.send(AgentEvent::Status("AI Responded".to_string()));
        Ok(())
    }

    async fn before_tool_call(&self, tool: &mut ToolCall) -> Result<()> {
        let _ = self.tx.send(AgentEvent::Status(format!("Executing Tool: {}", tool.name)));
        let _ = self.tx.send(AgentEvent::MetricUpdate {
            tokens: 0,
            context_size: 0,
            step: format!("Tool: {}", tool.name),
        });
        Ok(())
    }
}

pub struct AgentManager {
    agent: DeepAgent,
    event_tx: mpsc::UnboundedSender<AgentEvent>,
}

impl AgentManager {
    pub fn new(
        event_tx: mpsc::UnboundedSender<AgentEvent>,
        provider: Box<dyn ModelProvider>,
        embeddings: Arc<dyn EmbeddingProvider>,
        token_counter: Arc<dyn TokenCounter>,
        config: AppConfig,
    ) -> Result<Self> {
        let storage_root = PathBuf::from(&config.storage_path);
        let storage = FileStorage::new(storage_root.clone());
        
        let session_id = config.session_id.clone();
        let mp_config = config.to_mindpalace_config();
        
        let mut mp_middleware = MindPalaceMiddleware::hardened(
            storage.clone(),
            Arc::new(MockLlmClient),
            embeddings.clone(),
            token_counter.clone(),
            session_id.clone()
        );
        
        let brain = Arc::new(Brain::new(mp_config, None, Some(token_counter.clone())));
        mp_middleware.brain = brain.clone();

        let memory_controller = Arc::new(ResilientMemoryController::new(
            brain,
            storage.clone(),
            config.failure_threshold as usize
        ));

        let monitoring = MonitoringMiddleware {
            tx: event_tx.clone(),
            token_counter: token_counter.clone(),
        };

        let mut harness = Harness::new(provider);
        harness.add_middleware(Box::new(mp_middleware));
        harness.add_middleware(Box::new(monitoring));

        let exec_mode = match config.sandbox_mode.to_lowercase().as_str() {
            "docker" => ExecutionMode::Docker {
                image: config.docker_image.clone(),
                memory_limit: Some((config.ram_limit_mb * 1024 * 1024) as i64),
                cpu_quota: Some((config.cpu_limit_percent * 1000) as i64),
            },
            "wasm" => ExecutionMode::Wasm {
                module_path: config.wasm_module_path.map(PathBuf::from),
                mount_root: true,
                env_vars: config.wasm_env_vars.clone(),
            },
            _ => ExecutionMode::Local,
        };

        let vault_path = config.vault_path.map(PathBuf::from);
        let executor = mentalist::executor::SandboxedExecutor::new(
            exec_mode,
            std::env::current_dir()?,
            vault_path
        );

        let state_path = PathBuf::from(".agent/sessions").join(format!("session_{}.json", session_id));
        let state = if state_path.exists() {
            let data = std::fs::read_to_string(&state_path)?;
            let mut loaded_state: DeepAgentState = serde_json::from_str(&data)?;
            loaded_state.sandbox_root = std::env::current_dir()?;
            loaded_state
        } else {
            DeepAgentState {
                session_id,
                context: Context { items: vec![] },
                sandbox_root: std::env::current_dir()?,
            }
        };

        let agent = DeepAgent::new(harness, state, executor, memory_controller);

        Ok(Self { agent, event_tx })
    }

    pub async fn run_step(&mut self, input: String) -> Result<()> {
        if input.starts_with('/') {
            return self.handle_command(&input).await;
        }

        use mentalist::agent::AgentStepEvent;
        use futures_util::StreamExt;
        
        let mut stream = Box::pin(self.agent.step_stream(input));
        while let Some(res) = stream.next().await {
            match res? {
                AgentStepEvent::TextChunk(c) => {
                    let _ = self.event_tx.send(AgentEvent::TextChunk(c));
                }
                AgentStepEvent::Status(s) => {
                    let _ = self.event_tx.send(AgentEvent::Status(s));
                }
                AgentStepEvent::ToolStarted(t) => {
                    let _ = self.event_tx.send(AgentEvent::Status(format!("Tool: {}", t)));
                }
                AgentStepEvent::ToolFinished(t, _) => {
                    let _ = self.event_tx.send(AgentEvent::Status(format!("Finished Tool: {}", t)));
                }
            }
        }
        
        let _ = self.event_tx.send(AgentEvent::Status("Idle".to_string()));
        Ok(())
    }

    async fn handle_command(&mut self, input: &str) -> Result<()> {
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() { return Ok(()); }

        match parts[0] {
            "/session" => {
                if parts.len() < 2 {
                    let _ = self.event_tx.send(AgentEvent::Error("Usage: /session [list|switch <id>]".into()));
                    return Ok(());
                }
                match parts[1] {
                    "list" => {
                        let sessions_dir = PathBuf::from(".agent/sessions");
                        if !sessions_dir.exists() {
                            let _ = self.event_tx.send(AgentEvent::TextChunk("No sessions found.".into()));
                            return Ok(());
                        }
                        let entries = std::fs::read_dir(sessions_dir)?;
                        let mut list = String::from("Detected Sessions:\n");
                        for entry in entries {
                            let entry = entry?;
                            let name = entry.file_name().into_string().unwrap_or_default();
                            if name.ends_with(".json") {
                                let id = name.trim_start_matches("session_").trim_end_matches(".json");
                                list.push_str(&format!("- {}\n", id));
                            }
                        }
                        let _ = self.event_tx.send(AgentEvent::TextChunk(list));
                    }
                    "switch" => {
                        if parts.len() < 3 {
                            let _ = self.event_tx.send(AgentEvent::Error("Usage: /session switch <id>".into()));
                            return Ok(());
                        }
                        let new_id = parts[2];
                        let _ = self.event_tx.send(AgentEvent::Status(format!("Switching to {}...", new_id)));
                        
                        // We need a way to re-init. For now, we'll try to reload the state.
                        let state_path = PathBuf::from(".agent/sessions").join(format!("session_{}.json", new_id));
                        if !state_path.exists() {
                            let _ = self.event_tx.send(AgentEvent::Error(format!("Session {} not found", new_id)));
                            return Ok(());
                        }
                        
                        let data = std::fs::read_to_string(&state_path)?;
                        let mut new_state: DeepAgentState = serde_json::from_str(&data)?;
                        new_state.sandbox_root = std::env::current_dir()?;
                        self.agent.state = new_state;
                        
                        let _ = self.event_tx.send(AgentEvent::TextChunk(format!("Switched to session: {}\n", new_id)));
                    }
                    _ => {
                        let _ = self.event_tx.send(AgentEvent::Error("Usage: /session [list|switch <id>]".into()));
                    }
                }
            }
            "/consume" => {
                let _ = self.event_tx.send(AgentEvent::Status("Studying current directory...".into()));
                match self.consume_context().await {
                    Ok(count) => {
                        let _ = self.event_tx.send(AgentEvent::TextChunk(format!("Studied {} files. Knowledge base updated.", count)));
                    }
                    Err(e) => {
                        let _ = self.event_tx.send(AgentEvent::Error(format!("Consume failed: {}", e)));
                    }
                }
            }
            "/review" => {
                let _ = self.event_tx.send(AgentEvent::Status("Auditing staged changes...".into()));
                match self.review_vault().await {
                    Ok(report) => {
                        let _ = self.event_tx.send(AgentEvent::TextChunk(format!("## Audit Review Report\n{}", report)));
                    }
                    Err(e) => {
                        let _ = self.event_tx.send(AgentEvent::Error(format!("Review failed: {}", e)));
                    }
                }
            }
            _ => {
                let _ = self.event_tx.send(AgentEvent::Error(format!("Unknown command: {}", parts[0])));
            }
        }
        
        let _ = self.event_tx.send(AgentEvent::Status("Idle".to_string()));
        Ok(())
    }

    async fn consume_context(&mut self) -> Result<usize> {
        let root = std::env::current_dir()?;
        let mut count = 0;
        
        // Simple recursive text walker (ignoring binary/hidden/node_modules)
        let entries = self.walk_dir(&root)?;
        for path in entries {
            if let Ok(content) = std::fs::read_to_string(&path) {
                // For now, let's just trigger a specialized reasoning loop step
                let _ = self.agent.step(format!("Study this file and extract its core knowledge: {:?}\n\nCONTENT:\n{}", path.strip_prefix(&root).unwrap_or(&path), content)).await?;
                count += 1;
        }
        Ok(count)
    }

    fn walk_dir(&self, dir: &PathBuf) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                    if ["rs", "toml", "md", "txt", "js", "ts", "json", "env"].contains(&ext) && !name.starts_with('.') {
                        files.push(path);
                    }
                } else if path.is_dir() {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                    if name != "target" && name != "node_modules" && !name.starts_with('.') {
                        files.extend(self.walk_dir(&path)?);
                    }
                }
            }
        }
        Ok(files)
    }

    async fn review_vault(&mut self) -> Result<String> {
        let _ = self.event_tx.send(AgentEvent::Status("Reading vault contents...".into()));
        // Logic to read vault and run specialized AI step
        let report = self.agent.step("Perform an audit review of all files in the staging vault. Identify bugs, errors, and potential enhancements.".into()).await?;
        Ok(report)
    }
}

// Minimal Mock for initialization
struct MockLlmClient;
#[async_trait]
impl LlmClient for MockLlmClient {
    async fn completion(&self, _prompt: &str) -> Result<String> {
        Ok("Mock".to_string())
    }
}
