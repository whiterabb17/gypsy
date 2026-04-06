use std::sync::Arc;
use tokio::sync::mpsc;
use anyhow::Result;
use mentalist::{Harness, DeepAgent, Request, Response, ToolCall, ModelProvider};
use mentalist::executor::{ExecutionMode, MultiExecutor};
use mentalist::mcp::McpExecutor;
use mentalist::skills::SkillExecutor;
use mentalist::middleware::{Middleware, MindPalaceMiddleware, ToolDiscoveryMiddleware};
use mem_core::{FileStorage, EmbeddingProvider, TokenCounter, MemoryItem, MemoryRole};
use mem_resilience::ResilientMemoryController;
use async_trait::async_trait;
use crate::config::AppConfig;
use secrecy::ExposeSecret;

use crate::command_parser::{CommandParser, ToolArgumentParser};
use crate::context_consumer::ContextConsumer;
use crate::session::SessionManager;
use crate::error::{GypsyError, GypsyResult};
use std::path::PathBuf;

use shlex;
use chrono;
use futures_util::StreamExt;

#[derive(Debug, Clone)]
pub enum AgentEvent {
    Quit,
    Status(String),
    TextChunk(String),
    MetricUpdate {
        tokens: usize,
        input_tokens: usize,
        output_tokens: usize,
        context_size: usize,
        latency_ms: u128,
        step: String,
        tool_name: Option<String>,
    },
    Progress(f32),
    PhaseProgress(f32),
    ToolResult { name: String, success: bool },
    RequestFallback,
    Error(String),
}

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
            input_tokens: tokens,
            output_tokens: 0,
            context_size: req.context.items.len(),
            latency_ms: 0,
            step: "Thinking (LLM)".to_string(),
            tool_name: None,
        });
        let _ = self.tx.try_send(AgentEvent::Status("Thinking...".to_string()));
        Ok(())
    }

    async fn after_ai_call(&self, res: &mut Response) -> Result<()> {
        let tokens: usize = self.token_counter.count_tokens(&res.content);
        let _ = self.tx.try_send(AgentEvent::MetricUpdate {
            tokens,
            input_tokens: 0, // Ideally we'd get these from the response if available
            output_tokens: tokens,
            context_size: 0,
            latency_ms: 0, // Will be updated by AgentManager if timed
            step: "Thinking Complete".to_string(),
            tool_name: None,
        });
        let _ = self.tx.try_send(AgentEvent::Status("AI Responded".to_string()));
        Ok(())
    }

    async fn before_tool_call(&self, tool: &mut ToolCall) -> Result<()> {
        let _ = self.tx.try_send(AgentEvent::Status(format!("Executing Tool: {}", tool.name)));
        let _ = self.tx.try_send(AgentEvent::MetricUpdate {
            tokens: 0,
            input_tokens: 0,
            output_tokens: 0,
            context_size: 0,
            latency_ms: 0,
            step: format!("Tool: {}", tool.name),
            tool_name: Some(tool.name.clone()),
        });
        Ok(())
    }
}

pub struct AgentManager {
    pub agent: DeepAgent,
    pub multi_executor: Arc<MultiExecutor>,
    pub event_tx: mpsc::Sender<AgentEvent>,
    pub context_consumer: ContextConsumer,
    pub session_manager: SessionManager,
    pub config: AppConfig,
}

impl AgentManager {
    pub async fn new(
        event_tx: mpsc::Sender<AgentEvent>,
        provider: Arc<dyn ModelProvider>,
        embeddings: Arc<dyn EmbeddingProvider>,
        token_counter: Arc<dyn TokenCounter>,
        config: AppConfig,
    ) -> GypsyResult<Self> {
        let storage_root = PathBuf::from(&config.storage_path);
        let storage = FileStorage::new(storage_root.clone());
        
        let session_id = config.session_id.clone();
        let session_manager = SessionManager::new(&config.sessions_path);
        
        let vault_path = Self::get_vault_path(&config, &session_id);
        std::fs::create_dir_all(&vault_path).ok();

        let mp_middleware = MindPalaceMiddleware::hardened(
            storage.clone(),
            provider.clone(),
            embeddings.clone(),
            token_counter.clone(),
            session_id.clone(),
            config.embedding_dimension as usize,
            vault_path,
        );
        
        let brain = mp_middleware.brain.clone();
        tracing::info!("Brain initialized for session: {}", session_id);

        let memory_controller = Arc::new(ResilientMemoryController::new(
            brain.clone(),
            storage.clone(),
            config.failure_threshold as usize
        ));

        let monitoring = MonitoringMiddleware {
            tx: event_tx.clone(),
            token_counter: token_counter.clone(),
        };

        let mut harness = Harness::new(provider.clone());
        harness.add_middleware(Arc::new(mentalist::middleware::LoggingMiddleware));
        let mp_middleware = Arc::new(mp_middleware);
        harness.add_middleware(mp_middleware.clone());
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

        let vault_path_sandbox = config.vault_path.clone().map(PathBuf::from);
        let sandbox_executor = Arc::new(mentalist::executor::SandboxedExecutor::new(
            exec_mode,
            std::env::current_dir()?,
            vault_path_sandbox
        )?);

        let multi_executor = MultiExecutor::new();
        multi_executor.add_executor("sandbox".to_string(), sandbox_executor).await;

        for (name, cmd_line) in &config.mcp_servers {
            if let Some(parts) = shlex::split(cmd_line) {
                if !parts.is_empty() {
                    let cmd = parts[0].clone();
                    let args = parts[1..].to_vec();
                    tracing::info!("Registering MCP server: {} -> {} {:?}", name, cmd, args);
                    multi_executor.add_executor(format!("mcp:{}", name), Arc::new(McpExecutor::new(cmd, args))).await;
                }
            }
        }

        let fs_paths = if config.mcp_filesystem_paths.is_empty() {
            vec![".".to_string()]
        } else {
            config.mcp_filesystem_paths.iter()
                .filter(|p| PathBuf::from(p).exists())
                .cloned()
                .collect::<Vec<_>>()
        };
        multi_executor.add_executor("mcp:filesystem".to_string(), Arc::new(mentalist::mcp::BuiltinMcp::filesystem(fs_paths))).await;

        if let Some(api_key) = &config.firecrawl_api_key {
            multi_executor.add_executor("mcp:firecrawl".to_string(), Arc::new(mentalist::mcp::BuiltinMcp::firecrawl(api_key.expose_secret().clone()))).await;
        }

        let skills_path = PathBuf::from(&config.skills_path);
        if !skills_path.exists() {
            std::fs::create_dir_all(&skills_path).ok();
        }

        if let Ok(skill_executor) = SkillExecutor::new(skills_path.clone(), config.to_security_config()).await {
            multi_executor.add_executor("skills".to_string(), Arc::new(skill_executor)).await;
        }

        let multi_executor = Arc::new(multi_executor);
        harness.add_middleware(Arc::new(ToolDiscoveryMiddleware::new(multi_executor.clone())));

        let mut state = session_manager.load_session(&session_id)?;
        let mut current_ctx = (*state.context).clone();
        
        if let Some(prompt) = &config.system_prompt {
            if !current_ctx.items.iter().any(|i| i.role == MemoryRole::System && i.content.contains(prompt)) {
                current_ctx.items.insert(0, MemoryItem {
                    role: MemoryRole::System,
                    content: prompt.clone(),
                    timestamp: chrono::Utc::now().timestamp() as u64,
                    metadata: serde_json::json!({ "type": "system_prompt" }),
                });
            }
        }

        memory_controller.optimize_resilient(&mut current_ctx).await?;
        state.context = Arc::new(current_ctx);

        let scheduler = mp_middleware.dreamer.as_ref().map(|d| {
            let mut s = mem_dreamer::DreamScheduler::new(d.clone());
            s.start();
            s
        });

        let agent = DeepAgent::new(harness, state, multi_executor.clone(), memory_controller, scheduler);
        Ok(Self {
            agent,
            multi_executor,
            event_tx,
            context_consumer: ContextConsumer::new(),
            session_manager,
            config,
        })
    }

    pub async fn get_available_commands(&self) -> GypsyResult<Vec<String>> {
        let mut commands = vec![
            "/tools".to_string(),
            "/session".to_string(),
            "/consume".to_string(),
            "/review".to_string(),
            "/exit".to_string(),
            "/summarize".to_string(),
            "/mcp".to_string(),
            "/skills".to_string(),
        ];
        let tools = self.agent.executor.list_tools().await?;
        for t in tools {
            commands.push(format!("/{}", t.name));
        }
        
        Ok(commands)
    }

    pub async fn name(&self) -> &str { "Gypsy" }

    pub async fn run_step(&mut self, input: String) -> GypsyResult<()> {
        let start = std::time::Instant::now();
        if input.starts_with('/') {
            let res = self.handle_command(&input).await;
            let _ = self.save_current_session();
            return res;
        }

        use mentalist::agent::AgentStepEvent;
        
        let mut stream = Box::pin(self.agent.step_stream(input, self.config.to_agent_config()));
        let mut tool_results = Vec::new();
        
        while let Some(step_result) = stream.next().await {
            match step_result {
                Ok(event) => {
                    match event {
                        AgentStepEvent::TextChunk(c) => {
                            let _ = self.event_tx.try_send(AgentEvent::TextChunk(c));
                        }
                        AgentStepEvent::Status(s) => {
                            let _ = self.event_tx.try_send(AgentEvent::Status(s));
                        }
                        AgentStepEvent::ToolStarted(t) => {
                            let _ = self.event_tx.try_send(AgentEvent::Status(format!("Tool: {}", t)));
                        }
                        AgentStepEvent::ToolFinished(t, result) => {
                            tool_results.push((t.clone(), result.clone()));
                            let success = !result.to_lowercase().contains("error");
                            let _ = self.event_tx.try_send(AgentEvent::ToolResult { name: t.clone(), success });
                            let _ = self.event_tx.try_send(AgentEvent::Status(format!("Finished Tool: {}", t)));
                        }
                    }
                }
                Err(e) => {
                    let _ = self.event_tx.try_send(AgentEvent::Error(format!("Stream error: {}", e)));
                    break;
                }
            }
            tokio::task::yield_now().await;
        }
        
        drop(stream);

        let duration = start.elapsed().as_millis();
        let _ = self.event_tx.try_send(AgentEvent::MetricUpdate {
            tokens: 0,
            input_tokens: 0,
            output_tokens: 0,
            context_size: self.agent.state.context.items.len(),
            latency_ms: duration,
            step: "Step Complete".to_string(),
            tool_name: None,
        });

        // Commit accumulated tool results
        if !tool_results.is_empty() {
            let current_ctx = Arc::make_mut(&mut self.agent.state.context);
            for (t, result) in tool_results {
                current_ctx.items.push(MemoryItem {
                    role: MemoryRole::Tool,
                    content: result,
                    timestamp: chrono::Utc::now().timestamp() as u64,
                    metadata: serde_json::json!({"tool": t}),
                });
            }
            let _ = self.save_current_session();
        }
        
        let _ = self.save_current_session();
        let _ = self.event_tx.try_send(AgentEvent::Status("Idle".to_string()));
        Ok(())
    }

    async fn handle_command(&mut self, input: &str) -> GypsyResult<()> {
        let (command, args_str) = match CommandParser::parse(input) {
            Some(res) => res,
            None => return Ok(()),
        };

        match command {
            "/tools" => {
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
                        let ids = self.session_manager.list_sessions()?;
                        let mut list = String::from("Detected Sessions:\n");
                        for id in ids {
                            list.push_str(&format!("- {}\n", id));
                        }
                        let _ = self.event_tx.try_send(AgentEvent::TextChunk(list));
                    }
                    "switch" => {
                        if parts.len() < 2 {
                            let _ = self.event_tx.try_send(AgentEvent::Error("Usage: /session switch <id>".into()));
                            return Ok(());
                        }
                        let new_id = parts[1];
                        match self.session_manager.load_session(new_id) {
                            Ok(new_state) => {
                                self.agent.state = new_state;
                                let _ = self.event_tx.try_send(AgentEvent::TextChunk(format!("Switched to session: {}\n", new_id)));
                            }
                            Err(e) => {
                                let _ = self.event_tx.try_send(AgentEvent::Error(format!("Failed to load session: {}", e)));
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
            "/skills" => {
                let parts: Vec<&str> = args_str.split_whitespace().collect();
                if !parts.is_empty() && parts[0] == "reload" {
                    let _ = self.event_tx.try_send(AgentEvent::Status("Reloading skills...".into()));
                    let skills_path = PathBuf::from(&self.config.skills_path);
                    let security = self.config.to_security_config();
                    if let Ok(skill_executor) = SkillExecutor::new(skills_path, security).await {
                        self.multi_executor.add_executor("skills".to_string(), Arc::new(skill_executor)).await;
                        let _ = self.event_tx.try_send(AgentEvent::TextChunk("Skills reloaded successfully. 🚀\n".into()));
                    } else {
                        let _ = self.event_tx.try_send(AgentEvent::Error("Failed to reload skills.".into()));
                    }
                } else {
                    let _ = self.event_tx.try_send(AgentEvent::Error("Usage: /skills reload".into()));
                }
            }
            "/mcp" => {
                let parts: Vec<&str> = args_str.split_whitespace().collect();
                let sub_cmd = if parts.is_empty() { "list" } else { parts[0] };
                
                match sub_cmd {
                    "list" => {
                        let executors = self.multi_executor.list_executors().await;
                        let mut msg = String::from("### MCP Servers Status\n\n| Name | Status | State |\n| :--- | :--- | :--- |\n");
                        for (name, enabled, status) in executors {
                            let status_str = if enabled { "✅ Enabled" } else { "❌ Disabled" };
                            msg.push_str(&format!("| `{}` | {} | {} |\n", name, status_str, status));
                        }
                        let _ = self.event_tx.try_send(AgentEvent::TextChunk(msg));
                    }
                    "enable" | "disable" => {
                        if parts.len() < 2 {
                            let _ = self.event_tx.try_send(AgentEvent::Error(format!("Usage: /mcp {} <name>", sub_cmd)));
                            return Ok(());
                        }
                        let name = parts[1];
                        let enabled = sub_cmd == "enable";
                        if self.multi_executor.set_executor_enabled(name, enabled).await {
                            let _ = self.event_tx.try_send(AgentEvent::TextChunk(format!("Executor `{}` is now {}.", name, if enabled { "enabled" } else { "disabled" })));
                        } else {
                            let _ = self.event_tx.try_send(AgentEvent::Error(format!("Executor `{}` not found.", name)));
                        }
                    }
                    _ => {
                        let _ = self.event_tx.try_send(AgentEvent::Error("Usage: /mcp [list|enable <name>|disable <name>]".into()));
                    }
                }
            }
            "/summarize" => {
                let before = self.agent.state.context.items.len();
                let mut current_ctx = (*self.agent.state.context).clone();
                if let Err(e) = self.agent.harness.optimize_context(&mut current_ctx).await {
                    let _ = self.event_tx.try_send(AgentEvent::Error(format!("Summarization failed: {}", e)));
                } else {
                    self.agent.state.context = Arc::new(current_ctx);
                    let after = self.agent.state.context.items.len();
                    let _ = self.event_tx.try_send(AgentEvent::TextChunk(format!("Context optimized: {} -> {} items.\n", before, after)));
                    let _ = self.save_current_session();
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

    async fn execute_tool_command(&mut self, name: &str, args_str: &str, def: &mem_core::ToolDefinition) -> GypsyResult<()> {
        let args = ToolArgumentParser::parse(args_str, def)?;
        match self.agent.executor.execute(name, args.clone()).await {
            Ok(res) => {
                let current_ctx = Arc::make_mut(&mut self.agent.state.context);
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
                let _ = self.save_current_session();
                let success = !res.to_lowercase().contains("error");
                let _ = self.event_tx.try_send(AgentEvent::ToolResult { name: name.to_string(), success });
                let _ = self.event_tx.try_send(AgentEvent::TextChunk(format!("\n#### Tool Result: {}\n---\n{}\n---\n", name, res)));
            }
            Err(e) => {
                let _ = self.event_tx.try_send(AgentEvent::ToolResult { name: name.to_string(), success: false });
                let _ = self.event_tx.try_send(AgentEvent::Error(format!("Tool execution failed: {}", e)));
            }
        }
        Ok(())
    }

    async fn consume_context(&mut self) -> GypsyResult<usize> {
        let root = std::env::current_dir()?;
        let report = self.context_consumer.consume_with_limits(&root).await?;
        
        use futures_util::stream::{self, StreamExt};
        let harness = self.agent.harness.clone();
        let event_tx = self.event_tx.clone();
        let root_clone = root.clone();

        let items: Vec<PathBuf> = report.processed_files.into_iter().take(50).collect();
        let plan_total = items.len();
        let results: Vec<Option<(PathBuf, GypsyResult<mentalist::Response>)>> = stream::iter(items.into_iter().enumerate())
            .map(|(i, path)| {
                let harness = harness.clone();
                let event_tx = event_tx.clone();
                let root = root_clone.clone();
                async move {
                    let progress = (i + 1) as f32 / plan_total as f32;
                    let _ = event_tx.try_send(AgentEvent::Progress(progress));
                    let _ = event_tx.try_send(AgentEvent::PhaseProgress(progress));

                    if let Ok(content) = tokio::fs::read_to_string(&path).await {
                        let filename = path.strip_prefix(&root).unwrap_or(path.as_path()).to_path_buf();
                        let res = harness.run(mentalist::Request {
                            prompt: format!(
                                "Extract and summarize the key knowledge from this file for long-term memory. Be concise but thorough.\n\nFilename: {:?}\n\nCONTENT:\n{}", 
                                filename, content
                            ),
                            context: Arc::new(mem_core::Context { items: vec![] }),
                            tools: vec![],
                        }).await.map_err(|e| GypsyError::Mentalist(e.to_string()));
                        Some((filename, res))
                    } else {
                        None
                    }
                }
            })
            .buffer_unordered(3)
            .collect::<Vec<_>>()
            .await;

        let mut count = 0;
        let current_ctx = Arc::make_mut(&mut self.agent.state.context);
        for res in results.into_iter().flatten() {
            let (filename, summary_res) = res;
            match summary_res {
                Ok(summary) => {
                    current_ctx.items.push(MemoryItem {
                        role: MemoryRole::System,
                        content: format!("Knowledge extracted from {:?}:\n{}", filename, summary.content),
                        timestamp: chrono::Utc::now().timestamp() as u64,
                        metadata: serde_json::json!({"source": "consume", "file": filename}),
                    });
                    count += 1;
                }
                Err(e) => {
                    tracing::error!("Failed to summarize {:?}: {}", filename, e);
                }
            }
        }

        let _ = self.event_tx.try_send(AgentEvent::Progress(0.0));
        let _ = self.event_tx.try_send(AgentEvent::PhaseProgress(0.0));
        let _ = self.save_current_session();
        Ok(count)
    }

    async fn review_vault(&mut self) -> GypsyResult<String> {
        let vault_path = Self::get_vault_path(&self.config, &self.agent.state.session_id);
        if !vault_path.exists() {
            return Ok("Staging vault is empty.".to_string());
        }
        let mut vault_contents = String::new();
        let mut files_found = 0;
        let mut entries = std::fs::read_dir(&vault_path)?;
        while let Some(Ok(entry)) = entries.next() {
            if entry.path().is_file() {
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    files_found += 1;
                    vault_contents.push_str(&format!("### File: {}\n\n```\n{}\n```\n\n", entry.path().display(), content));
                }
            }
        }
        if files_found == 0 { return Ok("Staging vault is empty.".to_string()); }
        let report = self.agent.step(format!("Perform an audit review of these staged files in the vault:\n\n{}", vault_contents)).await?;
        Ok(report)
    }

    fn save_current_session(&self) -> GypsyResult<()> {
        let mgr = self.session_manager.clone();
        let state = self.agent.state.clone();
        tokio::spawn(async move {
            if let Err(e) = mgr.save_session(&state) {
                tracing::error!("Async session save failed: {}", e);
            }
        });
        Ok(())
    }

    pub fn get_vault_path(config: &AppConfig, session_id: &str) -> PathBuf {
        match config.vault_path.as_ref() {
            Some(p) => PathBuf::from(p),
            None => PathBuf::from(&config.sessions_path).join(session_id).join("vault"),
        }
    }
}
