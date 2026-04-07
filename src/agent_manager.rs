use crate::config::AppConfig;
use async_trait::async_trait;
use mem_core::{EmbeddingProvider, FileStorage, TokenCounter};
use mentalist::{
    executor::MultiExecutor, mcp::BuiltinMcp, middleware::Middleware,
    middleware::MindPalaceMiddleware, AgentRuntime, DefaultCritic, ExecutionLimits, Executor,
    MindPalaceLLM, MindPalaceMemory, MindPalacePlanner, ModelProvider, Policy, RuntimeEvent,
    SecurityEngine,
};
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use serde_json::Value;

// --- Builtin Tools ---

struct EchoTool;

#[async_trait]
impl mentalist::tools::Tool for EchoTool {
    fn schema(&self) -> mentalist::tools::ToolSchema {
        mentalist::tools::ToolSchema {
            name: "echo".into(),
            description: "Echoes back the message.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "msg": { "type": "string" }
                },
                "required": ["msg"]
            }),
            source: "builtin".into(),
        }
    }

    async fn execute(&self, input: Value) -> anyhow::Result<Value> {
        Ok(input)
    }
}

use crate::command_parser::CommandParser;
use crate::context_consumer::ContextConsumer;
use crate::error::GypsyResult;
use crate::prefs::PrefsManager;
use crate::session::SessionManager;
use std::path::PathBuf;
use tokio::sync::Mutex;

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
    ToolResult {
        name: String,
        success: bool,
    },
    AwaitingApproval(mem_planner::ExecutionPlan),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
    pub content: String,
}

pub struct AgentManager {
    pub runtime: Arc<AgentRuntime>,
    pub multi_executor: Arc<MultiExecutor>,
    pub mp_middleware: Arc<MindPalaceMiddleware>,
    pub discovered_skills: Vec<SkillInfo>,
    pub event_tx: mpsc::Sender<AgentEvent>,
    pub context_consumer: ContextConsumer,
    pub session_manager: Arc<SessionManager>,
    pub token_counter: Arc<dyn TokenCounter>,
    pub prefs: Arc<Mutex<PrefsManager>>,
    pub config: AppConfig,
    pub pending_approval: Arc<Mutex<Option<tokio::sync::mpsc::Sender<bool>>>>,
}

impl AgentManager {
    pub async fn new(
        event_tx: mpsc::Sender<AgentEvent>,
        provider: Arc<dyn ModelProvider>,
        embeddings: Arc<dyn EmbeddingProvider>,
        token_counter: Arc<dyn TokenCounter>,
        prefs: Arc<Mutex<PrefsManager>>,
        config: AppConfig,
    ) -> GypsyResult<Self> {
        let _ = event_tx
            .send(AgentEvent::Status("Initializing Gypsy...".into()))
            .await;

        let storage_root = PathBuf::from(&config.storage_path);
        let storage = FileStorage::new(storage_root.clone());

        let session_id = config.session_id.clone();
        let session_manager = Arc::new(SessionManager::new(&config.sessions_path));

        let vault_path = Self::get_vault_path(&config, &session_id);
        std::fs::create_dir_all(&vault_path).ok();

        // 1. Initialize Memory (MindPalace)
        let mp_config = config.to_mindpalace_config();
        let mp_middleware = mentalist::middleware::MindPalaceMiddleware::hardened(
            storage.clone(),
            provider.clone(),
            embeddings.clone(),
            token_counter.clone(),
            session_id.clone(),
            config.embedding_dimension,
            vault_path.clone(),
            Some(mp_config),
        );
        let brain = mp_middleware.brain.clone();

        // Correct MindPalaceMemory initialization with retriever
        let graph =
            Arc::new(mem_core::FactGraph::new(None).map_err(|e| anyhow::anyhow!("{:?}", e))?);
        let store = Arc::new(mem_retriever::InMemoryStore::default());
        let l_client = provider.clone() as Arc<dyn mem_core::LlmClient>;
        let retriever = mem_retriever::MemoryRetriever::new(
            storage.clone(),
            embeddings.clone(),
            l_client,
            store,
            graph,
        );
        let memory = Arc::new(MindPalaceMemory::new(brain, retriever));

        // 2. Initialize Executors & Tools
        let multi_executor = Arc::new(MultiExecutor::new());
        let mcp_root = PathBuf::from(&config.mcp_root_path);
        let mcp_timeout = std::time::Duration::from_secs(config.mcp_initialize_timeout_seconds);

        let _ = event_tx.send(AgentEvent::Status("Booting Agent...".into())).await;

        // Register Builtin Tools
        let _ = multi_executor.add_tool(Arc::new(EchoTool)).await;

        let fs_paths = if config.mcp_filesystem_paths.is_empty() {
            vec![std::env::current_dir()?.to_string_lossy().to_string()]
        } else {
            config
                .mcp_filesystem_paths
                .iter()
                .map(|p| PathBuf::from(p))
                .filter(|p| p.exists())
                .map(|p| p.to_string_lossy().to_string())
                .collect::<Vec<_>>()
        };

        // Parallel Initialization
        let mut spawn_handles = Vec::new();
        let prefs_guard = prefs.lock().await;

        if prefs_guard.is_mcp_enabled("filesystem") {
            let multi_executor = Arc::clone(&multi_executor);
            let mcp_root = mcp_root.clone();
            let event_tx = event_tx.clone();
            spawn_handles.push(tokio::spawn(async move {
                if std::env::var("GYPSY_SKIP_MCP_INSTALL").is_err() {
                    let _ = event_tx.send(AgentEvent::Status("Installing filesystem MCP...".into())).await;
                    let _ = BuiltinMcp::ensure_mcp_installed(&mcp_root, "@modelcontextprotocol/server-filesystem").await;
                }
                let _ = event_tx.send(AgentEvent::Status("Starting filesystem MCP...".into())).await;
                match BuiltinMcp::filesystem(fs_paths, Some(&mcp_root)) {
                    Ok(fs_mcp) => {
                        if let Err(e) = multi_executor
                            .add_executor(
                                "filesystem".to_string(),
                                Arc::new(fs_mcp.with_timeout(mcp_timeout)),
                            )
                            .await 
                        {
                            let _ = event_tx.send(AgentEvent::Error(format!("Filesystem MCP failed: {}", e))).await;
                        } else {
                            let _ = event_tx.send(AgentEvent::Status("Filesystem MCP Ready.".into())).await;
                        }
                    }
                    Err(e) => {
                        let _ = event_tx.send(AgentEvent::Error(format!("Filesystem MCP config error: {}", e))).await;
                    }
                }
            }));
        }

        if config.enable_ddg_search && prefs_guard.is_mcp_enabled("duckduckgo") {
            let multi_executor = Arc::clone(&multi_executor);
            let mcp_root = mcp_root.clone();
            let event_tx = event_tx.clone();
            spawn_handles.push(tokio::spawn(async move {
                if std::env::var("GYPSY_SKIP_MCP_INSTALL").is_err() {
                    let _ = event_tx.send(AgentEvent::Status("Installing search MCP...".into())).await;
                    let _ = BuiltinMcp::ensure_mcp_installed(&mcp_root, "duckduckgo-mcp-server").await;
                }
                let _ = event_tx.send(AgentEvent::Status("Starting search MCP...".into())).await;
                match BuiltinMcp::duckduckgo(Some(&mcp_root)) {
                    Ok(ddg_mcp) => {
                        if let Err(e) = multi_executor
                            .add_executor(
                                "duckduckgo".to_string(),
                                Arc::new(ddg_mcp.with_timeout(mcp_timeout)),
                            )
                            .await
                        {
                            let _ = event_tx.send(AgentEvent::Error(format!("Search MCP failed: {}", e))).await;
                        } else {
                            let _ = event_tx.send(AgentEvent::Status("Search MCP Ready.".into())).await;
                        }
                    }
                    Err(e) => {
                        let _ = event_tx.send(AgentEvent::Error(format!("Search MCP config error: {}", e))).await;
                    }
                }
            }));
        }

        if let Some(api_key) = config.firecrawl_api_key.as_ref() {
            if prefs_guard.is_mcp_enabled("firecrawl") {
                let multi_executor = Arc::clone(&multi_executor);
                let mcp_root = mcp_root.clone();
                let event_tx = event_tx.clone();
                let api_key = api_key.expose_secret().clone();
                spawn_handles.push(tokio::spawn(async move {
                    if std::env::var("GYPSY_SKIP_MCP_INSTALL").is_err() {
                        let _ = event_tx.send(AgentEvent::Status("Installing firecrawl MCP...".into())).await;
                        let _ = BuiltinMcp::ensure_mcp_installed(&mcp_root, "firecrawl-mcp").await;
                    }
                    let _ = event_tx.send(AgentEvent::Status("Starting firecrawl MCP...".into())).await;
                    match BuiltinMcp::firecrawl(api_key, Some(&mcp_root)) {
                        Ok(fc_mcp) => {
                            if let Err(e) = multi_executor
                                .add_executor(
                                    "firecrawl".to_string(),
                                    Arc::new(fc_mcp.with_timeout(mcp_timeout)),
                                )
                                .await
                            {
                                let _ = event_tx.send(AgentEvent::Error(format!("Firecrawl MCP failed: {}", e))).await;
                            } else {
                                let _ = event_tx.send(AgentEvent::Status("Firecrawl MCP Ready.".into())).await;
                            }
                        }
                        Err(e) => {
                            let _ = event_tx.send(AgentEvent::Error(format!("Firecrawl MCP config error: {}", e))).await;
                        }
                    }
                }));
            }
        }
        drop(prefs_guard);

        // MCPs finish initialization in parallel in the background
        // We no longer join_all(spawn_handles) here so startup is instant.

        let skills_path = PathBuf::from(&config.skills_path);
        if !skills_path.exists() {
            std::fs::create_dir_all(&skills_path).ok();
        }

        // 3. Assemble Cognitive Runtime
        // Use mem_planner::LlmPlanner wrapped by mentalist::MindPalacePlanner
        let planner_engine = Arc::new(mentalist::mem_planner::LlmPlanner::new(
            provider.clone()
        ));
        let planner = Arc::new(MindPalacePlanner::new(planner_engine));
        let executor = Arc::new(Executor::new(multi_executor.registry.clone()));
        let security = Arc::new(SecurityEngine::new(Policy::default()));

        // Wrap provider in mentalist bridge
        let llm = Arc::new(MindPalaceLLM::new(provider.clone()));
        let critic = Arc::new(LlmCritic::new(llm.clone()));

        let runtime = Arc::new(AgentRuntime {
            planner,
            executor,
            memory,
            llm,
            tools: multi_executor.registry.clone(),
            security,
            critic,
            limits: ExecutionLimits {
                max_steps: config.max_steps,
                timeout_seconds: config.mcp_initialize_timeout_seconds * 10,
            },
            middlewares: vec![Arc::new(mp_middleware.clone())],
        });

        let mut discovered_skills = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&skills_path) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let skill_md = entry.path().join("SKILL.md");
                    if skill_md.exists() {
                        if let Ok(content) = std::fs::read_to_string(&skill_md) {
                            if let Some((name, description)) = Self::parse_skill_metadata(&content) {
                                let folder_name = entry.path().file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
                                let meta = SkillInfo {
                                    name: folder_name.to_string(),
                                    description: description.clone(),
                                    path: entry.path(),
                                    content: content.clone(),
                                };
                                
                                // Register as a Mentalist Tool
                                let skill_name = meta.name.clone();
                                let skill_desc = meta.description.clone();
                                let instructions_clone = meta.content.clone();
                                
                                let skill_tool = mentalist::tools::Skill {
                                    name: skill_name,
                                    description: format!("Skill: {}. Use this to retrieve specialized instructions or workflows for this domain.", skill_desc),
                                    parameters: serde_json::json!({
                                        "type": "object",
                                        "properties": {
                                            "task_context": { "type": "string", "description": "Specific sub-task you need guidance on." }
                                        }
                                    }),
                                    handler: std::sync::Arc::new(move |_| {
                                        let instructions = instructions_clone.clone();
                                        Box::pin(async move {
                                            Ok(serde_json::json!({ "instructions": instructions }))
                                        })
                                    }),
                                };
                                
                                let _ = multi_executor.add_tool(std::sync::Arc::new(skill_tool));
                                discovered_skills.push(meta);
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            runtime,
            multi_executor,
            mp_middleware: Arc::new(mp_middleware),
            discovered_skills,
            event_tx,
            context_consumer: ContextConsumer::new(),
            session_manager,
            token_counter,
            prefs,
            config,
            pending_approval: Arc::new(Mutex::new(None)),
        })
    }

    fn parse_skill_metadata(content: &str) -> Option<(String, String)> {
        if !content.starts_with("---") {
            return None;
        }
        let tail = &content[3..];
        let end_idx = tail.find("---")?;
        let yaml_str = &tail[..end_idx];

        let mut name = None;
        let mut description = None;

        for line in yaml_str.lines() {
            if let Some((k, v)) = line.split_once(':') {
                match k.trim() {
                    "name" => {
                        name = Some(v.trim().trim_matches('"').trim_matches('\'').to_string())
                    }
                    "description" => {
                        description =
                            Some(v.trim().trim_matches('"').trim_matches('\'').to_string())
                    }
                    _ => {}
                }
            }
        }

        if let (Some(n), Some(d)) = (name, description) {
            Some((n, d))
        } else {
            None
        }
    }

    pub async fn get_available_commands(&self) -> GypsyResult<Vec<String>> {
        let mut commands = vec![
            "/tools".to_string(),
            "/session".to_string(),
            "/consume".to_string(),
            "/summarize".to_string(),
            "/skills".to_string(),
            "/review".to_string(),
            "/exit".to_string(),
            "/mcp".to_string(),
        ];
        let tools = self.runtime.tools.list_tools().await;
        for t in tools {
            commands.push(format!("/{}", t.name));
        }

        Ok(commands)
    }

    pub fn name(&self) -> &str {
        "Gypsy"
    }

    pub async fn run_step(&mut self, input: String) -> GypsyResult<()> {
        let start = std::time::Instant::now();
        if input.starts_with('/') {
            let res = self.handle_command(&input).await;
            let _ = self.save_current_session();
            return res;
        }

        let session_id = self.config.session_id.clone();
        let mut state = self.session_manager.load_session(&session_id)?;

        // Add user input to context BEFORE run
        let mut context = (*state.context).clone();
        context.items.push(mem_core::MemoryItem {
            role: mem_core::MemoryRole::User,
            content: input.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metadata: serde_json::json!({}),
        });
        state.context = Arc::new(context.clone());

        let (tx, mut rx) = mpsc::unbounded_channel();
        let (approve_tx, approve_rx) = mpsc::channel(1);
        let runtime = Arc::clone(&self.runtime);
        let event_tx = self.event_tx.clone();
        let input_clone = input.clone();
        let mut accumulated_response = String::new();
        
        {
            let mut guard = self.pending_approval.lock().await;
            *guard = Some(approve_tx);
        }

        tokio::spawn(async move {
            if let Err(e) = runtime.run(&input_clone, context, Some(tx), Some(approve_rx)).await {
                let _ = event_tx
                    .send(AgentEvent::Error(format!("Runtime error: {}", e)))
                    .await;
            }
        });

        while let Some(event) = rx.recv().await {
            match event {
                RuntimeEvent::AwaitingApproval(plan) => {
                    let _ = self.event_tx.send(AgentEvent::AwaitingApproval(plan)).await;
                }
                RuntimeEvent::Status(s) => {
                    let _ = self.event_tx.send(AgentEvent::Status(s)).await;
                }
                RuntimeEvent::TextChunk(c) => {
                    accumulated_response.push_str(&c);
                    let _ = self.event_tx.send(AgentEvent::TextChunk(c)).await;
                }
                RuntimeEvent::ToolStarted(t) => {
                    let _ = self
                        .event_tx
                        .send(AgentEvent::Status(format!("Tool: {}", t)))
                        .await;
                }
                RuntimeEvent::ToolFinished(t, _res, success) => {
                    let _ = self
                        .event_tx
                        .send(AgentEvent::ToolResult { name: t, success }).await;
                }
                RuntimeEvent::MetricUpdate { 
                    step, 
                    phase, 
                    input_tokens, 
                    output_tokens, 
                    context_size 
                } => {
                    let _ = self.event_tx.send(AgentEvent::MetricUpdate {
                        tokens: input_tokens + output_tokens,
                        input_tokens,
                        output_tokens,
                        context_size,
                        latency_ms: start.elapsed().as_millis(),
                        step: format!("Step {}: {}", step, phase),
                        tool_name: None,
                    }).await;
                }
            }
        }

        // Add assistant response to context AFTER run
        if !accumulated_response.is_empty() {
            let mut context = (*state.context).clone();
            context.items.push(mem_core::MemoryItem {
                role: mem_core::MemoryRole::Assistant,
                content: accumulated_response,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                metadata: serde_json::json!({}),
            });
            state.context = Arc::new(context);
        }

        self.session_manager.save_session(&state)?;
        let _ = self
            .event_tx
            .send(AgentEvent::Status("Idle".to_string())).await;
        Ok(())
    }

    pub async fn handle_command(&mut self, input: &str) -> GypsyResult<()> {
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        match parts[0] {
            "/approve" => {
                let mut guard = self.pending_approval.lock().await;
                if let Some(tx) = guard.take() {
                    let _ = tx.send(true).await;
                    let _ = self.event_tx.send(AgentEvent::Status("Plan approved. Resuming...".into())).await;
                } else {
                    let _ = self.event_tx.send(AgentEvent::Error("No pending plan to approve.".into())).await;
                }
            }
            "/cancel" | "/reject" => {
                let mut guard = self.pending_approval.lock().await;
                if let Some(tx) = guard.take() {
                    let _ = tx.send(false).await;
                    let _ = self.event_tx.send(AgentEvent::Status("Plan rejected.".into())).await;
                } else {
                    let _ = self.event_tx.send(AgentEvent::Error("No pending plan to reject.".into())).await;
                }
            }
            "/tools" => {
                let tools = self.runtime.tools.list_tools().await;
                let mut list = String::from("### Available Tools\n\n");
                for t in tools {
                    list.push_str(&format!("- `/{}`: {}\n", t.name, t.description));
                }
                let _ = self.event_tx.send(AgentEvent::TextChunk(list)).await;
            }
            "/session" => {
                let parts: Vec<&str> = args_str.split_whitespace().collect();
                if parts.is_empty() {
                    let _ = self.event_tx.send(AgentEvent::Error(
                        "Usage: /session [list|switch <id>]".into(),
                    )).await;
                    return Ok(());
                }
                match parts[0] {
                    "list" => {
                        let ids = self.session_manager.list_sessions()?;
                        let mut list = String::from("Detected Sessions:\n");
                        for id in ids {
                            list.push_str(&format!("- {}\n", id));
                        }
                        let _ = self.event_tx.send(AgentEvent::TextChunk(list)).await;
                    }
                    "switch" => {
                        if parts.len() < 2 {
                            let _ = self.event_tx.send(AgentEvent::Error(
                        "Usage: /session switch <id>".into(),
                    )).await;
                    return Ok(());
                }
                let new_id = parts[1];
                self.config.session_id = new_id.to_string();
                let _ = self.event_tx.send(AgentEvent::TextChunk(format!(
                    "Switched to session: {}\n",
                    new_id
                ))).await;
            }
            _ => {
                let _ = self.event_tx.send(AgentEvent::Error(
                    "Usage: /session [list|switch <id>]".into(),
                )).await;
            }
        }
    }
    "/consume" => {
        let _ = self
            .event_tx
            .send(AgentEvent::Status("Studying current directory...".into())).await;
        match self.consume_context().await {
            Ok(count) => {
                let _ = self.event_tx.send(AgentEvent::TextChunk(format!(
                    "Studied {} files. Knowledge base updated.",
                    count
                ))).await;
            }
            Err(e) => {
                let _ = self
                    .event_tx
                    .send(AgentEvent::Error(format!("Consume failed: {}", e))).await;
            }
        }
    }
    "/exit" => {
        let _ = self.event_tx.send(AgentEvent::Quit).await;
    }
            "/mcp" => {
                let parts: Vec<&str> = args_str.split_whitespace().collect();
                if parts.is_empty() || parts[0] == "list" {
                    let executors = self.multi_executor.registry.list_tools().await;
                    let disabled = self.prefs.lock().await.list_disabled_mcps();

                    let mut msg = String::from("### MCP Management\n\n");
                    msg.push_str("**Active Tools:**\n");
                    let mut seen_sources = std::collections::HashSet::new();
                    for t in executors {
                        let source = t.source.clone();
                        if !seen_sources.contains(&source) {
                            msg.push_str(&format!("- `{}` (Status: Active)\n", source));
                            seen_sources.insert(source);
                        }
                    }

                    if !disabled.is_empty() {
                        msg.push_str("\n**Disabled Servers:**\n");
                        for d in disabled {
                            msg.push_str(&format!("- `{}` (Status: Disabled)\n", d));
                        }
                    }

                    msg.push_str("\nUsage: `/mcp [list|enable <name>|disable <name>]`\n");
                    let _ = self.event_tx.send(AgentEvent::TextChunk(msg)).await;
                } else if parts[0] == "disable" && parts.len() > 1 {
                    let name = parts[1];
                    let mut prefs = self.prefs.lock().await;
                    if let Err(e) = prefs.set_mcp_enabled(name, false) {
                        let _ = self.event_tx.send(AgentEvent::Error(format!(
                            "Failed to persist choice: {}",
                            e
                        ))).await;
                    } else {
                        // Unload from runtime
                        self.multi_executor.unregister_executor(name).await;
                        let _ = self.event_tx.send(AgentEvent::TextChunk(format!(
                            "MCP `{}` has been disabled and persists across restarts. 🔒",
                            name
                        ))).await;
                    }
                } else if parts[0] == "enable" && parts.len() > 1 {
                    let name = parts[1];
                    let mut prefs = self.prefs.lock().await;
                    if let Err(e) = prefs.set_mcp_enabled(name, true) {
                        let _ = self.event_tx.send(AgentEvent::Error(format!(
                            "Failed to persist choice: {}",
                            e
                        ))).await;
                    } else {
                        let _ = self.event_tx.send(AgentEvent::TextChunk(format!("MCP `{}` has been enabled. Please restart Gypsy to fully reload it. 🔓", name))).await;
                    }
                }
            }
            "/summarize" => {
                let session_id = self.config.session_id.clone();
                let mut state = self.session_manager.load_session(&session_id)?;

                let _ = self
                    .event_tx
                    .send(AgentEvent::Status("Compressing context...".into())).await;

                let mut context = (*state.context).clone();
                if let Err(e) = self.mp_middleware.optimize_context(&mut context).await {
                    let _ = self
                        .event_tx
                        .send(AgentEvent::Error(format!("Summarization failed: {}", e))).await;
                } else {
                    let old_size = state.context.items.len();
                    state.context = Arc::new(context);
                    self.session_manager.save_session(&state)?;
                    let _ = self.event_tx.send(AgentEvent::TextChunk(format!(
                        "Context optimized. Items reduced from {} to {}.",
                        old_size,
                        state.context.items.len()
                    ))).await;
                }
            }
            "/skills" => {
                let json = serde_json::to_string_pretty(&self.discovered_skills)
                    .unwrap_or_else(|_| "[]".to_string());
                let _ = self.event_tx.send(AgentEvent::TextChunk(format!(
                    "### Discovered Skills (agentskills.io)\n\n```json\n{}\n```",
                    json
                ))).await;
            }
            _ if command.starts_with('/') => {
                let tool_name = &command[1..];
                let tools = self.runtime.tools.list_tools().await;
                if let Some(_tool_def) = tools.iter().find(|t| t.name == tool_name) {
                    let _ = self
                        .event_tx
                        .try_send(AgentEvent::Status(format!("Executing: {}", tool_name)));
                    // Manual call logic
                } else {
                    let _ = self.event_tx.try_send(AgentEvent::Error(format!(
                        "Unknown command or tool: {}",
                        command
                    )));
                }
            }
            _ => {
                let _ = self
                    .event_tx
                    .try_send(AgentEvent::Error(format!("Unknown command: {}", command)));
            }
        }
        let _ = self
            .event_tx
            .try_send(AgentEvent::Status("Idle".to_string()));
        Ok(())
    }

    pub async fn consume_context(&mut self) -> GypsyResult<usize> {
        let root = std::env::current_dir()?;
        let report = self.context_consumer.consume_with_limits(&root).await?;

        // 1. Ingest into Vector Memory (RAG)
        for (path, content) in &report.contents {
            let relative_path = path.strip_prefix(&root).unwrap_or(path);
            self.runtime
                .memory
                .store(mentalist::memory::MemoryEvent {
                    content: format!("File: {}\n---\n{}", relative_path.display(), content),
                    timestamp: chrono::Utc::now().timestamp() as u64,
                    metadata: serde_json::json!({
                        "source": "consume_context",
                        "path": relative_path.to_string_lossy(),
                        "project_root": root.to_string_lossy()
                    }),
                })
                .await
                .map_err(|e| crate::error::GypsyError::Mentalist(e.to_string()))?;
        }

        // 2. Add Project Summary to Working Context
        let session_id = self.config.session_id.clone();
        let mut state = self.session_manager.load_session(&session_id)?;
        let mut context = (*state.context).clone();

        let mut file_list = String::from("### Project Knowledge Bases (Ingested)\n\n");
        for path in &report.processed_files {
            let rel = path.strip_prefix(&root).unwrap_or(path);
            file_list.push_str(&format!("- `{}`\n", rel.display()));
        }

        context.items.push(mem_core::MemoryItem {
            role: mem_core::MemoryRole::System,
            content: format!(
                "I have ingested {} files from the current directory. Project root: {:?}\n\n{}",
                report.contents.len(),
                root,
                file_list
            ),
            timestamp: chrono::Utc::now().timestamp() as u64,
            metadata: serde_json::json!({"type": "knowledge_update"}),
        });

        state.context = Arc::new(context);
        self.session_manager.save_session(&state)?;

        Ok(report.processed_files.len())
    }

    pub fn save_current_session(&self) -> GypsyResult<()> {
        Ok(())
    }

    /// Shuts down the agent and its background processes gracefully.
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        self.runtime.shutdown().await
    }

    pub fn get_vault_path(config: &AppConfig, session_id: &str) -> PathBuf {
        match config.vault_path.as_ref() {
            Some(p) => PathBuf::from(p),
            None => PathBuf::from(&config.sessions_path)
                .join(session_id)
                .join("vault"),
        }
    }
}
