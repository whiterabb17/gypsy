use std::sync::Arc;
use tokio::sync::mpsc;
use anyhow::Result;
use mentalist::{Harness, DeepAgent, DeepAgentState, Request, Response, ToolCall, ModelProvider};
use mentalist::middleware::{Middleware, MindPalaceMiddleware};
use mem_core::{Context, FileStorage, EmbeddingProvider, LlmClient, TokenCounter};
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
}

// Minimal Mock for initialization
struct MockLlmClient;
#[async_trait]
impl LlmClient for MockLlmClient {
    async fn completion(&self, _prompt: &str) -> Result<String> {
        Ok("Mock".to_string())
    }
}
