use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use anyhow::Result;
use mentalist::{Harness, DeepAgent, DeepAgentState, Request, Response, ToolCall, ModelProvider};
use mentalist::executor::{ExecutionMode, MultiExecutor};
use mentalist::mcp::McpExecutor;
use mentalist::skills::SkillExecutor;
use mentalist::middleware::{Middleware, MindPalaceMiddleware, ToolDiscoveryMiddleware};
use mem_core::{Context, FileStorage, EmbeddingProvider, LlmClient, TokenCounter, MemoryItem, MemoryRole};
use mem_resilience::ResilientMemoryController;
use async_trait::async_trait;
use crate::config::AppConfig;

use crate::command_parser::{CommandParser, ToolArgumentParser};
use crate::context_consumer::ContextConsumer;
use std::path::PathBuf;
use std::time::Duration;
use brain::Brain;
use shlex;
use chrono;
use futures_util::StreamExt;

// --- Mocks & Utilities ---

pub struct MockLlmClient {
    pub responses: Vec<String>,
    call_count: Arc<Mutex<usize>>,
}

impl MockLlmClient {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub async fn get_call_count(&self) -> usize {
        *self.call_count.lock().await
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn completion(&self, _prompt: &str) -> Result<String> {
        let mut count = self.call_count.lock().await;
        let response = self.responses.get(*count)
            .cloned()
            .unwrap_or_else(|| "[]".to_string());
        *count += 1;
        Ok(response)
    }
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    Quit,
    Status(String),
    TextChunk(String),
    MetricUpdate {
        tokens: usize,
        context_size: usize,
        step: String,
    },
    Error(String),
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SessionFile {
    pub version: u32,
    pub state: DeepAgentState,
}

const CURRENT_SESSION_VERSION: u32 = 1;

pub struct MonitoringMiddleware {
    tx: mpsc::Sender<AgentEvent>,
    token_counter: Arc<dyn TokenCounter>,
}

#[async_trait]
impl Middleware for MonitoringMiddleware {
    fn name(&self) -> &str { "Monitoring" }

    async fn before_ai_call(&self, req: &mut Request) -> Result<()> {
        let tokens: usize = req.context.items.iter().map(|i| self.token_counter.count_tokens(&i.content)).sum();
        let _ = self.tx.try_send(AgentEvent::MetricUpdate {
            tokens,
            context_size: req.context.items.len(),
            step: "Thinking (LLM)".to_string(),
        });
        let _ = self.tx.try_send(AgentEvent::Status("Thinking...".to_string()));
        Ok(())
    }

    async fn after_ai_call(&self, _res: &mut Response) -> Result<()> {
        let _ = self.tx.try_send(AgentEvent::Status("AI Responded".to_string()));
        Ok(())
    }

    async fn before_tool_call(&self, tool: &mut ToolCall) -> Result<()> {
        let _ = self.tx.try_send(AgentEvent::Status(format!("Executing Tool: {}", tool.name)));
        let _ = self.tx.try_send(AgentEvent::MetricUpdate {
            tokens: 0,
            context_size: 0,
            step: format!("Tool: {}", tool.name),
        });
        Ok(())
    }
}

pub struct AgentManager {
    pub agent: DeepAgent,
    pub event_tx: mpsc::Sender<AgentEvent>,
    pub context_consumer: ContextConsumer,
    pub config: AppConfig,
}

impl AgentManager {
    pub async fn new(
        event_tx: mpsc::Sender<AgentEvent>,
        provider: Arc<dyn ModelProvider>,
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
            Arc::new(MockLlmClient::new(vec!["[]".to_string()])),
            embeddings.clone(),
            token_counter.clone(),
            session_id.clone(),
            config.embedding_dimension,
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
        harness.add_middleware(Arc::new(mentalist::middleware::LoggingMiddleware));
        harness.add_middleware(Arc::new(mp_middleware));
        harness.add_middleware(Arc::new(monitoring));

        let exec_mode = match config.sandbox_mode.to_lowercase().as_str() {
            "docker" => ExecutionMode::Docker {
                image: config.docker_image.clone(),
                memory_limit: Some((config.ram_limit_mb * 1024 * 1024) as i64),
                cpu_quota: Some((config.cpu_limit_percent * 1000) as i64),
            },
            "wasm" => ExecutionMode::Wasm {
                module_path: config.wasm_module_path.clone().map(PathBuf::from),
                mount_root: true,
                env_vars: config.wasm_env_vars.clone(),
            },
            _ => ExecutionMode::Local,
        };

        let vault_path = config.vault_path.clone().map(PathBuf::from);
        let sandbox_executor = Arc::new(mentalist::executor::SandboxedExecutor::new(
            exec_mode,
            std::env::current_dir()?,
            vault_path
        )?);

        let mut multi_executor = MultiExecutor::new();
        multi_executor.add_executor(sandbox_executor);

        for (_name, cmd_line) in &config.mcp_servers {
            let parts: Vec<String> = shlex::split(cmd_line)
                .unwrap_or_default();
            if !parts.is_empty() {
                let cmd = parts[0].clone();
                let args = parts[1..].to_vec();
                multi_executor.add_executor(Arc::new(McpExecutor::new(cmd, args)));
            }
        }

        let fs_paths = if config.mcp_filesystem_paths.is_empty() {
            vec![".".to_string()]
        } else {
            config.mcp_filesystem_paths.clone()
        };
        multi_executor.add_executor(Arc::new(mentalist::mcp::BuiltinMcp::filesystem(fs_paths)));

        if let Some(ref api_key) = config.firecrawl_api_key {
            multi_executor.add_executor(Arc::new(mentalist::mcp::BuiltinMcp::firecrawl(api_key.clone())));
        }

        let skills_path = PathBuf::from(&config.skills_path);
        let skill_executor = tokio::select! {
            res = SkillExecutor::new(skills_path.clone()) => match res {
                Ok(e) => e,
                Err(err) => {
                    tracing::warn!("Skill executor failed to load from {:?}: {}. Using empty.", skills_path, err);
                    SkillExecutor { skills_root: skills_path, skills: std::collections::HashMap::new() }
                }
            },
            _ = tokio::time::sleep(Duration::from_secs(10)) => {
                tracing::warn!("Skill executor initialization timeout, continuing with empty skills");
                SkillExecutor { skills_root: skills_path, skills: std::collections::HashMap::new() }
            }
        };
        let skill_executor = Arc::new(skill_executor);
        multi_executor.add_executor(skill_executor);

        let multi_executor = Arc::new(multi_executor);
        harness.add_middleware(Arc::new(ToolDiscoveryMiddleware::new(multi_executor.clone())));

        let sessions_dir = PathBuf::from(&config.sessions_path);
        if !sessions_dir.exists() {
            let _ = std::fs::create_dir_all(&sessions_dir);
        }

        let state_path = sessions_dir.join(format!("session_{}.json", session_id));
        let state = if state_path.exists() {
            match std::fs::read_to_string(&state_path) {
                Ok(data) => {
                    match serde_json::from_str::<SessionFile>(&data) {
                        Ok(mut session) => {
                            tracing::info!("Loaded session version {} from {}", session.version, state_path.display());
                            // Migration logic can be added here
                            session.state.sandbox_root = std::env::current_dir()?;
                            session.state
                        }
                        Err(parse_err) => {
                            // Fallback to legacy or handle corruption
                            if let Ok(mut legacy_state) = serde_json::from_str::<DeepAgentState>(&data) {
                                tracing::info!("Migrating legacy session to version {}", CURRENT_SESSION_VERSION);
                                legacy_state.sandbox_root = std::env::current_dir()?;
                                legacy_state
                            } else {
                                tracing::warn!("Failed to parse session {}: {}. Creating backup and new session.", state_path.display(), parse_err);
                                let backup_path = state_path.with_extension("json.corrupt");
                                let _ = std::fs::copy(&state_path, &backup_path);
                                DeepAgentState {
                                    session_id,
                                    context: Arc::new(Context { items: vec![] }),
                                    sandbox_root: std::env::current_dir()?,
                                }
                            }
                        }
                    }
                }
                Err(read_err) => {
                    tracing::error!("Failed to read session file: {}. Creating new.", read_err);
                    DeepAgentState {
                        session_id,
                        context: Arc::new(Context { items: vec![] }),
                        sandbox_root: std::env::current_dir()?,
                    }
                }
            }
        } else {
            DeepAgentState {
                session_id,
                context: Arc::new(Context { items: vec![] }),
                sandbox_root: std::env::current_dir()?,
            }
        };

        let agent = DeepAgent::new(harness, state, multi_executor, memory_controller);
        Ok(Self {
            agent,
            event_tx,
            context_consumer: ContextConsumer::new(),
            config,
        })
    }

    pub async fn get_available_commands(&self) -> Result<Vec<String>> {
        let mut commands = vec![
            "/tools".to_string(),
            "/session".to_string(),
            "/consume".to_string(),
            "/review".to_string(),
            "/exit".to_string(),
            "/summarize".to_string(),
        ];
        
        let tools = self.agent.executor.list_tools().await?;
        for t in tools {
            commands.push(format!("/{}", t.name));
        }
        
        Ok(commands)
    }

    pub async fn name(&self) -> &str { "Gypsy" }

    pub async fn run_step(&mut self, input: String) -> Result<()> {
        if input.starts_with('/') {
            let res = self.handle_command(&input).await;
            let _ = self.save_current_session();
            return res;
        }

        use mentalist::agent::AgentStepEvent;
        
        let mut stream = Box::pin(self.agent.step_stream(input, mentalist::agent::StepConfig::default()));
        let mut total_chunks = 0;
        let mut last_status = "Processing...".to_string();

        while let Some(step_result) = stream.next().await {
            match step_result {
                Ok(event) => {
                    match event {
                        AgentStepEvent::TextChunk(c) => {
                            total_chunks += 1;
                            let _ = self.event_tx.try_send(AgentEvent::TextChunk(c));
                        }
                        AgentStepEvent::Status(s) => {
                            last_status = s.clone();
                            let _ = self.event_tx.try_send(AgentEvent::Status(s));
                        }
                        AgentStepEvent::ToolStarted(t) => {
                            tracing::debug!("Tool started: {}", t);
                            let _ = self.event_tx.try_send(AgentEvent::Status(format!("Tool: {}", t)));
                        }
                        AgentStepEvent::ToolFinished(t, result) => {
                            tracing::debug!("Tool finished: {} -> {} bytes", t, result.len());
                            let _ = self.event_tx.try_send(AgentEvent::Status(format!("Finished Tool: {}", t)));
                        }
                    }
                }
                Err(e) => {
                    let error_msg = format!("Stream error after {} chunks at status '{}': {}", total_chunks, last_status, e);
                    tracing::error!("{}", error_msg);
                    let _ = self.event_tx.try_send(AgentEvent::Error(error_msg));
                    break;
                }
            }
            // Simple backpressure throttle
            tokio::task::yield_now().await;
        }
        
        drop(stream); // Release borrow on self.agent for session saving
        let _ = self.save_current_session();
        let _ = self.event_tx.try_send(AgentEvent::Status("Idle".to_string()));
        Ok(())
    }

    async fn handle_command(&mut self, input: &str) -> Result<()> {
        let (command, args_str) = match CommandParser::parse(input) {
            Some(res) => res,
            None => return Ok(()),
        };

        match command {
            "/tools" => {
                let _ = self.event_tx.try_send(AgentEvent::Status("Discovering tools...".into()));
                let tools = self.agent.executor.list_tools().await?;
                let mut list = String::from("### Available Tools\n\n");
                for t in tools {
                    list.push_str(&format!("- `/{}`: {}\n", t.name, t.description));
                }
                let _ = self.event_tx.try_send(AgentEvent::TextChunk(list));
            }
            "/session" => {
                let parts: Vec<&str> = args_str.split_whitespace().collect();
                if parts.is_empty() {
                    let _ = self.event_tx.try_send(AgentEvent::Error("Usage: /session [list|switch <id>]".into()));
                    return Ok(());
                }
                match parts[0] {
                    "list" => {
                        let sessions_dir = PathBuf::from(".agent/sessions");
                        if !sessions_dir.exists() {
                            let _ = self.event_tx.try_send(AgentEvent::TextChunk("No sessions found.".into()));
                            return Ok(());
                        }
                        let entries = std::fs::read_dir(sessions_dir)?;
                        let mut list = String::from("Detected Sessions:\n");
                        for entry in entries {
                            let entry = entry?;
                            let name = entry.file_name().into_string().unwrap_or_default();
                            if name.ends_with(".json") || name.ends_with(".session") {
                                let id = name.trim_start_matches("session_").trim_end_matches(".json").trim_end_matches(".session");
                                list.push_str(&format!("- {}\n", id));
                            }
                        }
                        let _ = self.event_tx.try_send(AgentEvent::TextChunk(list));
                    }
                    "switch" => {
                        if parts.len() < 2 {
                            let _ = self.event_tx.try_send(AgentEvent::Error("Usage: /session switch <id>".into()));
                            return Ok(());
                        }
                        let new_id = parts[1];
                        let _ = self.event_tx.try_send(AgentEvent::Status(format!("Switching to {}...", new_id)));
                        
                        let sessions_path = self.config.sessions_path.clone();
                        
                        let session_file = format!("session_{}.json", new_id);
                        let mut state_path = PathBuf::from(&sessions_path).join(&session_file);
                        if !state_path.exists() {
                             state_path = PathBuf::from(&sessions_path).join(format!("session_{}.session", new_id));
                        }

                        if !state_path.exists() {
                            let _ = self.event_tx.try_send(AgentEvent::Error(format!("Session {} not found", new_id)));
                            return Ok(());
                        }
                        
                        match std::fs::read_to_string(&state_path) {
                            Ok(data) => {
                                match serde_json::from_str::<SessionFile>(&data) {
                                    Ok(mut session) => {
                                        session.state.sandbox_root = std::env::current_dir()?;
                                        self.agent.state = session.state;
                                        let _ = self.event_tx.try_send(AgentEvent::TextChunk(format!("Switched to session: {}\n", new_id)));
                                    }
                                    Err(_) => {
                                        // Try legacy
                                        match serde_json::from_str::<DeepAgentState>(&data) {
                                            Ok(mut new_state) => {
                                                new_state.sandbox_root = std::env::current_dir()?;
                                                self.agent.state = new_state;
                                                let _ = self.event_tx.try_send(AgentEvent::TextChunk(format!("Switched to session: {} (legacy format)\n", new_id)));
                                            }
                                            Err(e) => {
                                                let _ = self.event_tx.try_send(AgentEvent::Error(format!("Failed to parse session state: {}", e)));
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = self.event_tx.try_send(AgentEvent::Error(format!("Failed to read session file: {}", e)));
                            }
                        }
                    }
                    _ => {
                        let _ = self.event_tx.try_send(AgentEvent::Error("Usage: /session [list|switch <id>]".into()));
                    }
                }
            }
            "/consume" => {
                let _ = self.event_tx.try_send(AgentEvent::Status("Studying current directory...".into()));
                match self.consume_context().await {
                    Ok(count) => {
                        let _ = self.event_tx.try_send(AgentEvent::TextChunk(format!("Studied {} files. Knowledge base updated.", count)));
                    }
                    Err(e) => {
                        let _ = self.event_tx.try_send(AgentEvent::Error(format!("Consume failed: {}", e)));
                    }
                }
            }
            "/review" => {
                let _ = self.event_tx.try_send(AgentEvent::Status("Auditing staged changes...".into()));
                match self.review_vault().await {
                    Ok(report) => {
                        let _ = self.event_tx.try_send(AgentEvent::TextChunk(format!("## Audit Review Report\n{}", report)));
                    }
                    Err(e) => {
                        let _ = self.event_tx.try_send(AgentEvent::Error(format!("Review failed: {}", e)));
                    }
                }
            }
            "/exit" => {
                let _ = self.event_tx.try_send(AgentEvent::Quit);
            }
            "/summarize" => {
                let _ = self.event_tx.try_send(AgentEvent::Status("Optimizing context...".into()));
                let before = self.agent.state.context.items.len();
                
                let mut current_ctx = (*self.agent.state.context).clone();
                if let Err(e) = self.agent.harness.optimize_context(&mut current_ctx).await {
                    let _ = self.event_tx.try_send(AgentEvent::Error(format!("Summarization failed: {}", e)));
                } else {
                    self.agent.state.context = Arc::new(current_ctx);
                    let after = self.agent.state.context.items.len();
                    let _ = self.event_tx.try_send(AgentEvent::TextChunk(format!("Context optimized: {} -> {} items.\n", before, after)));
                }
            }
            _ if command.starts_with('/') => {
                let tool_name = &command[1..];
                let tools = self.agent.executor.list_tools().await?;
                if let Some(tool_def) = tools.iter().find(|t| t.name == tool_name).cloned() {
                    self.execute_tool_command(tool_name, args_str, &tool_def).await?;
                } else {
                    let _ = self.event_tx.try_send(AgentEvent::Error(format!("Unknown command or tool: {}", command)));
                }
            }
            _ => {
                let _ = self.event_tx.try_send(AgentEvent::Error(format!("Unknown command: {}", command)));
            }
        }
        
        let _ = self.event_tx.try_send(AgentEvent::Status("Idle".to_string()));
        Ok(())
    }

    async fn execute_tool_command(&mut self, name: &str, args_str: &str, def: &mem_core::ToolDefinition) -> Result<()> {
        let _ = self.event_tx.try_send(AgentEvent::Status(format!("Executing Tool: {}", name)));
        
        let args = ToolArgumentParser::parse(args_str, def)?;
        
        match self.agent.executor.execute(name, args.clone()).await {
            Ok(res) => {
                let mut current_ctx = (*self.agent.state.context).clone();
                current_ctx.items.push(MemoryItem {
                    role: MemoryRole::User,
                    content: format!("MANUAL TOOL CALL: /{} {}", name, args_str),
                    timestamp: chrono::Utc::now().timestamp() as u64,
                    metadata: serde_json::json!({}),
                });
                current_ctx.items.push(MemoryItem {
                    role: MemoryRole::Tool,
                    content: res.clone(),
                    timestamp: chrono::Utc::now().timestamp() as u64,
                    metadata: serde_json::json!({"tool": name}),
                });
                self.agent.state.context = Arc::new(current_ctx);
                
                let _ = self.event_tx.try_send(AgentEvent::TextChunk(format!("\n#### Tool Result: {}\n---\n{}\n---\n", name, res)));
            }
            Err(e) => {
                let _ = self.event_tx.try_send(AgentEvent::Error(format!("Tool execution failed: {}", e)));
            }
        }
        Ok(())
    }


    async fn consume_context(&mut self) -> Result<usize> {
        let root = std::env::current_dir()?;
        let report = self.context_consumer.consume_with_limits(&root).await?;
        
        let mut count = 0;
        for path in report.processed_files {
            if let Ok(content) = tokio::fs::read_to_string(&path).await {
                let filename = path.strip_prefix(&root).unwrap_or(&path);
                let _ = self.agent.step(format!("Study this file and extract its core knowledge: {:?}\n\nCONTENT:\n{}", filename, content)).await?;
                count += 1;
            }
        }
        Ok(count)
    }

    async fn review_vault(&mut self) -> Result<String> {
        let _ = self.event_tx.try_send(AgentEvent::Status("Reading vault contents...".into()));
        let report = self.agent.step("Perform an audit review of all files in the staging vault. Identify bugs, errors, and potential enhancements.".into()).await?;
        Ok(report)
    }

    pub fn save_session(&self, path: PathBuf) -> Result<()> {
        let session = SessionFile {
            version: CURRENT_SESSION_VERSION,
            state: self.agent.state.clone(),
        };
        let data = serde_json::to_string_pretty(&session)?;
        
        let mut temp_path = path.clone();
        temp_path.set_extension("tmp");
        
        std::fs::write(&temp_path, data)?;
        std::fs::rename(temp_path, path)?;
        Ok(())
    }

    fn save_current_session(&self) -> Result<()> {
        let path = PathBuf::from(&self.config.sessions_path).join(format!("session_{}.json", self.agent.state.session_id));
        self.save_session(path)
    }
}
