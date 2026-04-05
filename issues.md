CRITICAL BUGS & ISSUES
1. UNSAFE POINTER MANIPULATION IN TESTS (CRITICAL)
File: tests/agent_manager_tests.rs (lines 89-92) Severity: 🔴 CRITICAL - UB Risk

Rust
// CURRENT (DANGEROUS):
unsafe {
    let ptr = &manager.agent.state as *const DeepAgentState as *mut DeepAgentState;
    (*ptr).context = Arc::new(current_ctx);
}
Issue: This violates Rust's borrowing rules and causes undefined behavior. The Arc pointer is being cast to mutable and modified while the original reference exists.

Fix:

Rust
// ROBUST FIX:
// Option 1: Use interior mutability pattern
#[derive(Clone)]
pub struct DeepAgentStateWrapper {
    inner: Arc<RwLock<DeepAgentState>>,
}

// Option 2: Return new state from mutations
pub fn update_context(&self, new_context: Arc<Context>) -> DeepAgentState {
    let mut new_state = self.clone();
    new_state.context = new_context;
    new_state
}

// In test:
let new_state = manager.agent.state.update_context(Arc::new(current_ctx));
// Then update manager
2. RACE CONDITION IN LOG MANAGEMENT (HIGH)
File: src/main.rs (lines 316-320) Severity: 🟠 HIGH

Rust
// CURRENT (FLAWED):
if state.log.len() > 1000 {
    state.log.drain(0..state.log.len() - 1000);  // ⚠️ Off-by-one potential
}
Issue: Under high concurrency, logs can exceed 1000 indefinitely. No rate limiting on consumer side.

Fix:

Rust
// ROBUST FIX:
const MAX_LOG_SIZE: usize = 1000;
const DRAIN_THRESHOLD: usize = 50;

if state.log.len() > MAX_LOG_SIZE + DRAIN_THRESHOLD {
    let drain_count = state.log.len() - MAX_LOG_SIZE;
    state.log.drain(0..drain_count);
    tracing::warn!("Drained {} old logs to maintain memory", drain_count);
}

// Optional: Add backpressure to MPSC channel
let mut log_rx = log_rx.with_capacity(100);  // Bounded channel
3. UNHANDLED OLLAMA PROCESS SPAWNING (HIGH)
File: src/main.rs (lines 70-108) Severity: 🟠 HIGH - Resource Leak

Rust
// CURRENT (INCOMPLETE):
async fn start_ollama_daemon() -> Result<()> {
    let mut cmd = tokio::process::Command::new(path);
    if path.contains("Resources") || path.contains("bin") {
        cmd.arg("serve");
    }
    if cmd.spawn().is_ok() {   // ⚠️ Spawned but never tracked!
        return Ok(());
    }
}
Issue: Child process is spawned but never stored or managed. No cleanup on exit. Resource leak.

Fix:

Rust
// ROBUST FIX:
use std::sync::{Arc, Mutex};

static OLLAMA_CHILD: once_cell::sync::Lazy<Arc<Mutex<Option<tokio::process::Child>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(None)));

async fn start_ollama_daemon() -> Result<()> {
    for path in paths {
        if std::path::Path::new(path).exists() {
            let mut cmd = tokio::process::Command::new(path);
            if path.contains("Resources") || path.contains("bin") {
                cmd.arg("serve");
            }
            match cmd.spawn() {
                Ok(child) => {
                    let mut guard = OLLAMA_CHILD.lock().unwrap();
                    *guard = Some(child);
                    tracing::info!("Ollama daemon started with PID");
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!("Failed to start ollama from {}: {}", path, e);
                    continue;
                }
            }
        }
    }
    Err(anyhow::anyhow!("Could not start Ollama from any known path"))
}

// Cleanup on app exit:
fn cleanup_ollama() {
    if let Ok(mut guard) = OLLAMA_CHILD.lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            tracing::info!("Ollama daemon terminated");
        }
    }
}
4. PROVIDER INSTANTIATION DUPLICATION (MEDIUM)
File: src/main.rs (lines 118-189) Severity: 🟡 MEDIUM - Memory Waste

Rust
// CURRENT (DUPLICATED):
match config.provider.to_lowercase().as_str() {
    "anthropic" => {
        let provider = Arc::new(mem_core::AnthropicProvider::new(...));
        Ok(ProviderStack {
            model: Box::new(mem_core::AnthropicProvider::new(...)),  // ⚠️ DUPLICATE
            embeddings: provider.clone(),
            token_counter: provider,
        })
    }
    // ... repeated for each provider
}
Issue: Each provider is instantiated twice unnecessarily, doubling memory and connection overhead.

Fix:

Rust
// ROBUST FIX:
fn create_providers(config: &AppConfig) -> Result<ProviderStack> {
    let (model, embeddings, token_counter): (
        Box<dyn mentalist::ModelProvider>,
        Arc<dyn mem_core::EmbeddingProvider>,
        Arc<dyn mem_core::TokenCounter>,
    ) = match config.provider.to_lowercase().as_str() {
        "anthropic" => {
            let provider = Arc::new(mem_core::AnthropicProvider::new(
                config.anthropic_api_key.as_ref().context("ANTHROPIC_API_KEY missing")?.clone(),
                config.model_name.clone(),
            ));
            (
                Box::new(provider.clone()),
                provider.clone() as Arc<dyn mem_core::EmbeddingProvider>,
                provider as Arc<dyn mem_core::TokenCounter>,
            )
        }
        // ... similar for other providers
        _ => {
            let provider = Arc::new(mem_core::OllamaProvider::new(
                config.model_name.clone(),
                config.embedding_model.clone(),
            ));
            (
                Box::new(provider.clone()),
                provider.clone(),
                provider,
            )
        }
    };
    
    Ok(ProviderStack { model, embeddings, token_counter })
}
5. UNCHECKED TOKIO RUNTIME DEADLOCK (HIGH)
File: src/agent_manager.rs (lines 164-168) Severity: 🟠 HIGH - Potential Deadlock

Rust
// CURRENT (RISKY):
let skill_executor = tokio::task::block_in_place(|| {
    tokio::runtime::Handle::current().block_on(async {
        SkillExecutor::new(skills_path).await
    })
})?;
Issue: block_in_place within async context can cause thread starvation if not on a blocking-aware runtime thread. If called from non-worker thread, will panic.

Fix:

Rust
// ROBUST FIX:
// In AgentManager::new() - make it async if not already
pub async fn new_async(
    event_tx: mpsc::UnboundedSender<AgentEvent>,
    provider: Box<dyn ModelProvider>,
    embeddings: Arc<dyn EmbeddingProvider>,
    token_counter: Arc<dyn TokenCounter>,
    config: AppConfig,
) -> Result<Self> {
    // ... existing code ...
    
    // Non-blocking initialization with timeout
    let skill_executor = tokio::select! {
        result = SkillExecutor::new(skills_path.clone()) => result?,
        _ = tokio::time::sleep(Duration::from_secs(10)) => {
            tracing::warn!("Skill executor initialization timeout, continuing without skills");
            SkillExecutor::new_empty()?
        }
    };
    
    // ... rest of code ...
}

// Or in sync context (main):
let skill_executor = tokio::spawn(async {
    SkillExecutor::new(skills_path).await
}).await??;
6. AUTOCOMPLETE BUFFER OVERFLOW (MEDIUM)
File: src/main.rs (lines 381-405) Severity: 🟡 MEDIUM - DoS Vector

Rust
// CURRENT (UNBOUNDED):
KeyCode::Char(c) => {
    state.input_buffer.push(c);  // ⚠️ No limit!
    if state.input_buffer.starts_with('/') {
        state.autocomplete_suggestions = state
            .available_commands
            .iter()
            .filter(|cmd| cmd.starts_with(&state.input_buffer))
            .cloned()  // ⚠️ Allocates each frame
            .collect();
    }
}
Issue: Input buffer has no size limit. Can consume unbounded memory if user pastes large text.

Fix:

Rust
// ROBUST FIX:
const MAX_INPUT_LEN: usize = 4096;
const MAX_SUGGESTIONS: usize = 10;

KeyCode::Char(c) => {
    if state.input_buffer.len() < MAX_INPUT_LEN {
        state.input_buffer.push(c);
        
        if state.input_buffer.starts_with('/') {
            state.autocomplete_suggestions = state
                .available_commands
                .iter()
                .filter(|cmd| cmd.starts_with(&state.input_buffer))
                .take(MAX_SUGGESTIONS)  // Cap suggestions
                .cloned()
                .collect();
        } else {
            state.autocomplete_suggestions.clear();
        }
    } else {
        tracing::warn!("Input buffer limit exceeded");
    }
}
7. SESSION STATE SERIALIZATION ERRORS (HIGH)
File: src/agent_manager.rs (lines 177-189) Severity: 🟠 HIGH - Silent Failures

Rust
// CURRENT (UNGUARDED):
let state = if state_path.exists() {
    let data = std::fs::read_to_string(&state_path)?;
    let mut loaded_state: DeepAgentState = serde_json::from_str(&data)?;  // ⚠️ Silent corrupt state
    loaded_state.sandbox_root = std::env::current_dir()?;
    loaded_state
} else {
    // ... create new state
};
Issue: If session file is corrupted, it fails without recovery. No validation, no backup.

Fix:

Rust
// ROBUST FIX:
let state = if state_path.exists() {
    match std::fs::read_to_string(&state_path) {
        Ok(data) => {
            match serde_json::from_str::<DeepAgentState>(&data) {
                Ok(mut loaded_state) => {
                    tracing::info!("Loaded session from {}", state_path.display());
                    loaded_state.sandbox_root = std::env::current_dir()?;
                    loaded_state
                }
                Err(parse_err) => {
                    tracing::warn!(
                        "Failed to parse session {}: {}. Creating backup and new session.",
                        state_path.display(),
                        parse_err
                    );
                    
                    // Backup corrupted file
                    let backup_path = state_path.with_extension("json.corrupt");
                    std::fs::copy(&state_path, &backup_path)
                        .ok_or_else(|| anyhow::anyhow!("Could not backup corrupted session"))?;
                    
                    // Create fresh state
                    DeepAgentState {
                        session_id: session_id.clone(),
                        context: Arc::new(Context { items: vec![] }),
                        sandbox_root: std::env::current_dir()?,
                    }
                }
            }
        }
        Err(read_err) => {
            tracing::error!("Failed to read session file: {}. Creating new.", read_err);
            DeepAgentState {
                session_id: session_id.clone(),
                context: Arc::new(Context { items: vec![] }),
                sandbox_root: std::env::current_dir()?,
            }
        }
    }
} else {
    // ... create new
};
8. COMMAND PARSING INJECTION VECTOR (MEDIUM)
File: src/agent_manager.rs (lines 243-252) Severity: 🟡 MEDIUM - Command Injection

Rust
// CURRENT (NAIVE):
let parts: Vec<&str> = input.split_whitespace().collect();
if parts.is_empty() { return Ok(()); }
let command = parts[0];
let args_str = if input.len() > command.len() {
    input[command.len()..].trim()  // ⚠️ Indirect slicing risk
} else {
    ""
};
Issue: String slicing on UTF-8 boundaries can cause panic if input contains multi-byte chars.

Fix:

Rust
// ROBUST FIX:
let input_trimmed = input.trim();
let parts: Vec<&str> = input_trimmed.split_whitespace().collect();
if parts.is_empty() { 
    return Ok(()); 
}

let command = parts[0];
let args_start = input_trimmed.find(&command)? + command.len();
let args_str = input_trimmed.get(args_start..)
    .map(|s| s.trim())
    .unwrap_or("");

// Or use proper parsing:
let mut chars = input_trimmed.chars();
let cmd: String = chars.by_ref().take_while(|&c| !c.is_whitespace()).collect();
let args = chars.collect::<String>().trim().to_string();
9. MODAL LOG ENTRY ENUM MISMATCH (MEDIUM)
File: src/ui.rs (lines 73-94) Severity: 🟡 MEDIUM - Logic Error

Rust
// CURRENT (INCOMPLETE):
let chat_log: Vec<&LogEntry> = state.log.iter()
    .filter(|e| {
        match e {
            LogEntry::User(_) | LogEntry::Gypsy(_) | LogEntry::Error(_) => true,
            _ => false
        }
    })
    .collect();
Issue: The pattern matching in rendering (lines 98-112) doesn't cover all LogEntry variants returned by other systems. If a Debug or Warn somehow enters chat_log, it will silently skip.

Fix:

Rust
// ROBUST FIX:
let chat_log: Vec<&LogEntry> = state.log.iter()
    .filter(|e| matches!(e, LogEntry::User(_) | LogEntry::Gypsy(_) | LogEntry::Error(_)))
    .collect();

let mut chat_text = Text::default();
for entry in chat_log {
    let line = match entry {
        LogEntry::User(msg) => Line::from(vec![
            Span::styled("User: ", Style::default().fg(Color::Cyan)),
            Span::raw(msg),
        ]),
        LogEntry::Gypsy(msg) => Line::from(vec![
            Span::styled("Gypsy: ", Style::default().fg(Color::Green)),
            Span::raw(msg),
        ]),
        LogEntry::Error(msg) => Line::from(vec![
            Span::styled("ERROR: ", Style::default().fg(Color::Red)),
            Span::raw(msg),
        ]),
        // Add explicit unreachable for exhaustiveness checking
        _ => unreachable!("Chat log should only contain User/Gypsy/Error"),
    };
    chat_text.lines.push(line);
}
10. MISSING ERROR CONTEXT IN STREAMING (MEDIUM)
File: src/agent_manager.rs (lines 220-237) Severity: 🟡 MEDIUM

Rust
// CURRENT (SILENT FAILURES):
let mut stream = Box::pin(self.agent.step_stream(input, mentalist::agent::StepConfig::default()));
while let Some(res) = stream.next().await {
    match res? {  // ⚠️ Errors silently propagate
        AgentStepEvent::TextChunk(c) => {
            let _ = self.event_tx.send(AgentEvent::TextChunk(c));
        }
        // ...
    }
}
Issue: Stream errors propagate without context about which step failed. UI can't show meaningful error.

Fix:

Rust
// ROBUST FIX:
let mut stream = Box::pin(self.agent.step_stream(input, mentalist::agent::StepConfig::default()));
let mut total_chunks = 0;
let mut last_status = "Processing...".to_string();

while let Some(step_result) = stream.next().await {
    match step_result {
        Ok(event) => {
            match event {
                AgentStepEvent::TextChunk(c) => {
                    total_chunks += 1;
                    let _ = self.event_tx.send(AgentEvent::TextChunk(c));
                }
                AgentStepEvent::Status(s) => {
                    last_status = s.clone();
                    let _ = self.event_tx.send(AgentEvent::Status(s));
                }
                AgentStepEvent::ToolStarted(t) => {
                    tracing::debug!("Tool started: {}", t);
                    let _ = self.event_tx.send(AgentEvent::Status(format!("Tool: {}", t)));
                }
                AgentStepEvent::ToolFinished(t, result) => {
                    tracing::debug!("Tool finished: {} -> {} bytes", t, result.len());
                    let _ = self.event_tx.send(AgentEvent::Status(format!("Finished Tool: {}", t)));
                }
            }
        }
        Err(e) => {
            let error_msg = format!(
                "Stream error after {} chunks at status '{}': {}",
                total_chunks, last_status, e
            );
            tracing::error!("{}", error_msg);
            let _ = self.event_tx.send(AgentEvent::Error(error_msg));
            break;
        }
    }
}

let _ = self.event_tx.send(AgentEvent::Status("Idle".to_string()));
11. FILE WALK SYMLINK INFINITE LOOP (MEDIUM)
File: src/agent_manager.rs (lines 460-481) Severity: 🟡 MEDIUM - DoS Vector

Rust
// CURRENT (NO SYMLINK HANDLING):
fn walk_dir(&self, dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                // ... add file
            } else if path.is_dir() {  // ⚠️ Symlinks to dirs will loop!
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                if name != "target" && name != "node_modules" && !name.starts_with('.') {
                    files.extend(self.walk_dir(&path)?);  // Infinite loop risk
                }
            }
        }
    }
    Ok(files)
}
Issue: Symlinks or circular directory structures cause infinite recursion/DoS.

Fix:

Rust
// ROBUST FIX:
use std::collections::HashSet;

pub fn walk_dir_safe(&self, start_dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut visited = HashSet::new();
    self._walk_dir_recursive(start_dir, &mut files, &mut visited, 0)?;
    Ok(files)
}

fn _walk_dir_recursive(
    &self,
    dir: &PathBuf,
    files: &mut Vec<PathBuf>,
    visited: &mut HashSet<std::fs::Metadata>,
    depth: usize,
) -> Result<()> {
    const MAX_DEPTH: usize = 20;
    const MAX_FILES: usize = 10000;
    
    if depth > MAX_DEPTH {
        tracing::warn!("Max directory depth reached at {}", dir.display());
        return Ok(());
    }
    
    if files.len() > MAX_FILES {
        tracing::warn!("Max file count reached in directory walk");
        return Ok(());
    }
    
    if dir.is_dir() {
        // Get metadata to detect cycles via inode
        match std::fs::metadata(dir) {
            Ok(meta) => {
                if visited.contains(&meta) {
                    tracing::debug!("Skipping visited directory: {}", dir.display());
                    return Ok(());
                }
                visited.insert(meta);
            }
            Err(e) => {
                tracing::warn!("Cannot stat {}: {}", dir.display(), e);
                return Ok(());
            }
        }
        
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                if ["rs", "toml", "md", "txt", "js", "ts", "json", "env"].contains(&ext) 
                    && !name.starts_with('.') 
                {
                    files.push(path);
                }
            } else if path.is_dir() && !path.is_symlink() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                if name != "target" && name != "node_modules" && !name.starts_with('.') {
                    self._walk_dir_recursive(&path, files, visited, depth + 1)?;
                }
            }
        }
    }
    Ok(())
}
12. EVENT CHANNEL UNBOUNDED SENDER (MEDIUM)
File: src/main.rs (line 270) & src/agent_manager.rs (line 32) Severity: 🟡 MEDIUM - Memory Leak Under Load

Rust
// CURRENT (UNBOUNDED):
let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AgentEvent>();
let (log_tx, mut log_rx) = mpsc::unbounded_channel::<LogEntry>();
Issue: Unbounded channels can grow without limit if producer outpaces consumer, consuming all memory.

Fix:

Rust
// ROBUST FIX:
const EVENT_CHANNEL_CAPACITY: usize = 1000;
const LOG_CHANNEL_CAPACITY: usize = 5000;

let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(EVENT_CHANNEL_CAPACITY);
let (log_tx, mut log_rx) = mpsc::channel::<LogEntry>(LOG_CHANNEL_CAPACITY);

// When sending, handle backpressure:
match event_tx.try_send(event) {
    Ok(_) => {},
    Err(mpsc::error::TrySendError::Full(_)) => {
        tracing::warn!("Event channel full, dropping event");
    }
    Err(mpsc::error::TrySendError::Closed(_)) => {
        tracing::error!("Event channel closed");
        break;
    }
}
⚙️ ARCHITECTURAL BOTTLENECKS
Bottleneck 1: Monolithic AgentManager
File: src/agent_manager.rs (entire file)

The AgentManager mixes concerns:

State management
Command parsing
Tool discovery
Context consumption
Vault reviewing
Solution: Split into separate managers

Rust
pub struct AgentManager {
    agent: DeepAgent,
    event_tx: mpsc::UnboundedSender<AgentEvent>,
}

pub struct CommandParser;
pub struct ContextConsumer;
pub struct VaultAuditor;

impl CommandParser {
    fn parse(&self, input: &str) -> Result<Command>;
}

impl ContextConsumer {
    async fn consume(&self, root: &Path) -> Result<usize>;
}
Bottleneck 2: Synchronous Config Loading in Async Context
File: src/main.rs (line 261)

AppConfig::from_env() is synchronous but called in async context during Tokio initialization phase.

Solution:

Rust
// Use a lazy_static or once_cell
use once_cell::sync::Lazy;

static CONFIG: Lazy<AppConfig> = Lazy::new(|| AppConfig::from_env());

// In main, just reference it:
let config = &*CONFIG;
Bottleneck 3: Memory Unbounded Log Vector
File: src/main.rs (lines 300-330)

State log is a simple Vec with hardcoded drain at 1000 items.

Solution: Use a ring buffer

Rust
use ringbuffer::RingBuffer;

pub struct AppState {
    pub log: RingBuffer<LogEntry>,
    // ...
}

impl AppState {
    pub fn new() -> Self {
        Self {
            log: RingBuffer::new(1000),
            // ...
        }
    }
}

// Auto-size management:
pub fn add_log(&mut self, entry: LogEntry) {
    self.log.push_back(entry);  // Auto-evicts oldest
}
Bottleneck 4: No Stream Backpressure in Step Execution
File: src/agent_manager.rs (lines 220-240)

The stream consumes all events without applying backpressure, can exhaust memory if agent generates data faster than UI processes.

Solution:

Rust
// Use bounded streams with rate limiting
use futures_util::stream::StreamExt;

let mut stream = Box::pin(self.agent.step_stream(input, config));
let mut rate_limiter = RateLimiter::new(100);  // 100 events per 100ms

while let Some(res) = stream.next().await {
    rate_limiter.acquire().await;  // Throttle
    match res? {
        // ... handle event
    }
}
Bottleneck 5: No Connection Pooling for LLM Requests
File: src/main.rs (line 25) & src/agent_manager.rs

New reqwest::Client created for each Ollama check.

Solution:

Rust
static HTTP_CLIENT: once_cell::sync::Lazy<reqwest::Client> = 
    once_cell::sync::Lazy::new(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(4)
            .build()
            .unwrap()
    });

// Usage:
let health_url = format!("{}/api/tags", config.ollama_base_url);
HTTP_CLIENT.get(&health_url).send().await?;
Bottleneck 6: Linear Command Lookup
File: src/agent_manager.rs (lines 357-364)

Commands are looked up with .iter().find(), O(n) for each command.

Solution:

Rust
use std::collections::HashMap;

pub struct CommandRegistry {
    commands: HashMap<String, CommandDef>,
}

impl CommandRegistry {
    pub fn get(&self, name: &str) -> Option<&CommandDef> {
        self.commands.get(name)
    }
}
Bottleneck 7: Synchronous File I/O in Event Loop
File: src/agent_manager.rs (lines 444-457)

consume_context calls std::fs::read_to_string synchronously in async context for potentially gigabytes of files.

Solution:

Rust
async fn consume_context(&mut self, limit: usize) -> Result<usize> {
    let root = std::env::current_dir()?;
    let mut count = 0;
    
    let entries = self.walk_dir(&root)?;
    let mut tasks = vec![];
    
    for path in entries.into_iter().take(limit) {
        let agent_clone = self.agent.clone();  // If cloneable
        tasks.push(tokio::task::spawn(async move {
            if let Ok(content) = tokio::fs::read_to_string(&path).await {
                agent_clone.step(format!("Study: {}", path.display())).await
            } else {
                Ok(())
            }
        }));
    }
    
    for task in tasks {
        task.await??;
        count += 1;
    }
    
    Ok(count)
}
Bottleneck 8: Hardcoded File Extensions
File: src/agent_manager.rs (lines 469)

Rust
if ["rs", "toml", "md", "txt", "js", "ts", "json", "env"].contains(&ext) && ...
Hardcoded list prevents extensibility.

Solution:

Rust
pub struct FileWalkerConfig {
    allowed_extensions: HashSet<String>,
    excluded_dirs: HashSet<String>,
    max_file_size: usize,
}

impl Default for FileWalkerConfig {
    fn default() -> Self {
        let mut allowed = HashSet::new();
        allowed.extend(["rs", "toml", "md", "txt", "js", "ts", "json", "env", "py", "go"]
            .iter().map(|s| s.to_string()));
        
        let mut excluded = HashSet::new();
        excluded.extend(["target", "node_modules", ".git", "__pycache__"]
            .iter().map(|s| s.to_string()));
        
        Self {
            allowed_extensions: allowed,
            excluded_dirs: excluded,
            max_file_size: 10 * 1024 * 1024,  // 10MB
        }
    }
}

fn walk_dir_with_config(&self, dir: &Path, config: &FileWalkerConfig) -> Result<Vec<PathBuf>> {
    // Use config
}
🔧 SIMPLIFIED IMPLEMENTATIONS NEEDING ROBUSTNESS
1. MockLlmClient is Too Simple
File: src/main.rs (lines 191-197) & src/agent_manager.rs (lines 491-498)

Rust
// CURRENT (USELESS):
struct MockLlmClient;
#[async_trait::async_trait]
impl mem_core::LlmClient for MockLlmClient {
    async fn completion(&self, _prompt: &str) -> Result<String> {
        Ok("[]".to_string())
    }
}
Issue: Mock doesn't test any real behavior.

Fix:

Rust
#[cfg(test)]
mod mocks {
    use super::*;

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
        async fn completion(&self, prompt: &str) -> Result<String> {
            let mut count = self.call_count.lock().await;
            let response = self.responses.get(*count)
                .cloned()
                .unwrap_or_else(|| "No more responses".to_string());
            *count += 1;
            Ok(response)
        }
    }
}
2. Tool Argument Parsing is Naive
File: src/agent_manager.rs (lines 408-441)

Sequential fallback: JSON → key=value → single param

Fix:

Rust
pub struct ToolArgumentParser;

impl ToolArgumentParser {
    pub fn parse(
        args_str: &str,
        def: &mem_core::ToolDefinition,
    ) -> Result<serde_json::Value> {
        // 1. Try JSON with validation against schema
        if let Ok(val) = serde_json::from_str(args_str) {
            self.validate_against_schema(&val, def)?;
            return Ok(val);
        }

        // 2. Try key=value with type coercion
        let mut map = serde_json::Map::new();
        let props = def.parameters
            .get("properties")
            .and_then(|p| p.as_object())
            .ok_or_else(|| anyhow::anyhow!("No schema properties"))?;

        let pairs = shlex::split(args_str)
            .ok_or_else(|| anyhow::anyhow!("Invalid shell syntax"))?;

        for pair in pairs {
            if let Some((k, v)) = pair.split_once('=') {
                if let Some(param_schema) = props.get(k) {
                    let typed_val = self.coerce_type(v, param_schema)?;
                    map.insert(k.to_string(), typed_val);
                } else {
                    return Err(anyhow::anyhow!("Unknown parameter: {}", k));
                }
            }
        }

        if !map.is_empty() {
            return Ok(serde_json::Value::Object(map));
        }

        // 3. Single param positional
        if props.len() == 1 {
            let key = props.keys().next().unwrap();
            let param_schema = &props[key];
            let typed_val = self.coerce_type(args_str, param_schema)?;
            return Ok(serde_json::json!({ key: typed_val }));
        }

        Err(anyhow::anyhow!("Could not parse arguments"))
    }

    fn coerce_type(&self, val_str: &str, schema: &serde_json::Value) -> Result<serde_json::Value> {
        let type_str = schema.get("type").and_then(|t| t.as_str()).unwrap_or("string");
        match type_str {
            "integer" => val_str.parse::<i64>().map(|v| serde_json::json!(v))
                .map_err(|_| anyhow::anyhow!("Invalid integer")),
            "number" => val_str.parse::<f64>().map(|v| serde_json::json!(v))
                .map_err(|_| anyhow::anyhow!("Invalid number")),
            "boolean" => {
                match val_str.to_lowercase().as_str() {
                    "true" | "yes" | "1" => Ok(serde_json::json!(true)),
                    "false" | "no" | "0" => Ok(serde_json::json!(false)),
                    _ => Err(anyhow::anyhow!("Invalid boolean")),
                }
            }
            _ => Ok(serde_json::json!(val_str)),
        }
    }

    fn validate_against_schema(
        &self,
        val: &serde_json::Value,
        def: &mem_core::ToolDefinition,
    ) -> Result<()> {
        // Use jsonschema crate for validation
        let schema = serde_json::to_value(&def.parameters)?;
        jsonschema::validate(val, &schema)
            .map_err(|e| anyhow::anyhow!("Schema validation failed: {}", e))
    }
}
3. Session Management Lacks Versioning
File: src/agent_manager.rs (lines 177-189)

No migration path when session schema changes.

Fix:

Rust
#[derive(Serialize, Deserialize, Debug)]
pub struct SessionFile {
    pub version: u32,
    pub state: DeepAgentState,
    pub metadata: SessionMetadata,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SessionMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub description: Option<String>,
}

const CURRENT_SESSION_VERSION: u32 = 1;

fn load_session(path: &Path) -> Result<DeepAgentState> {
    let data = std::fs::read_to_string(path)?;
    let session_file: SessionFile = serde_json::from_str(&data)?;
    
    match session_file.version {
        1 => Ok(session_file.state),
        _ => Err(anyhow::anyhow!(
            "Unsupported session version: {}",
            session_file.version
        )),
    }
}

fn save_session(path: &Path, state: &DeepAgentState) -> Result<()> {
    let session_file = SessionFile {
        version: CURRENT_SESSION_VERSION,
        state: state.clone(),
        metadata: SessionMetadata {
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            description: None,
        },
    };
    let json = serde_json::to_string_pretty(&session_file)?;
    std::fs::write(path, json)?;
    Ok(())
}
4. Consume/Review Commands Lack Validation
File: src/agent_manager.rs (lines 319-339, 483-488)

No limits on what /consume processes or /review returns.

Fix:

Rust
pub struct ContextConsumerConfig {
    pub max_files: usize,
    pub max_file_size: u64,
    pub max_total_bytes: u64,
    pub timeout_per_file: Duration,
}

pub struct ContextConsumer {
    config: ContextConsumerConfig,
}

impl ContextConsumer {
    pub async fn consume_with_limits(&self, root: &Path) -> Result<ConsumeReport> {
        let mut report = ConsumeReport::default();
        let mut total_bytes = 0u64;
        
        let entries = self.walk_dir(root)?;
        
        for path in entries.iter().take(self.config.max_files) {
            if total_bytes > self.config.max_total_bytes {
                report.truncated = true;
                break;
            }
            
            match tokio::time::timeout(
                self.config.timeout_per_file,
                tokio::fs::read_to_string(&path),
            ).await {
                Ok(Ok(content)) => {
                    let bytes = content.len() as u64;
                    if bytes > self.config.max_file_size {
                        report.skipped_oversized.push(path.clone());
                        continue;
                    }
                    
                    total_bytes += bytes;
                    report.processed_files.push(path.clone());
                    report.total_bytes += bytes;
                }
                Ok(Err(e)) => {
                    tracing::warn!("Failed to read {}: {}", path.display(), e);
                    report.read_errors.push((path.clone(), e.to_string()));
                }
                Err(_) => {
                    tracing::warn!("Timeout reading {}", path.display());
                    report.timeouts.push(path.clone());
                }
            }
        }
        
        Ok(report)
    }
}

#[derive(Debug, Default)]
pub struct ConsumeReport {
    pub processed_files: Vec<PathBuf>,
    pub skipped_oversized: Vec<PathBuf>,
    pub read_errors: Vec<(PathBuf, String)>,
    pub timeouts: Vec<PathBuf>,
    pub total_bytes: u64,
    pub truncated: bool,
}
5. UI Rendering Lacks Pagination
File: src/ui.rs (lines 96-157)

Large logs are rendered entirely, causing performance issues.

Fix:

Rust
pub struct LogPaginator {
    pub visible_items: Vec<LogEntry>,
    pub total_items: usize,
    pub page: usize,
    pub items_per_page: usize,
}

impl LogPaginator {
    pub fn new(items_per_page: usize) -> Self {
        Self {
            visible_items: vec![],
            total_items: 0,
            page: 0,
            items_per_page,
        }
    }

    pub fn update(&mut self, all_items: &[LogEntry]) {
        self.total_items = all_items.len();
        let start = self.page * self.items_per_page;
        let end = std::cmp::min(start + self.items_per_page, all_items.len());
        self.visible_items = all_items[start..end].to_vec();
    }

    pub fn next_page(&mut self) {
        if (self.page + 1) * self.items_per_page < self.total_items {
            self.page += 1;
        }
    }

    pub fn prev_page(&mut self) {
        if self.page > 0 {
            self.page -= 1;
        }
    }
}

// In render:
paginator.update(&state.log);
for entry in &paginator.visible_items {
    // Render...
}
6. Error Handling Uses Generic Result
File: Throughout

All error types use anyhow::Result without custom error types.

Fix:

Rust
#[derive(Debug, thiserror::Error)]
pub enum GypsyError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Agent error: {0}")]
    AgentError(String),
    
    #[error("Tool execution error: {0}")]
    ToolError(String),
    
    #[error("Session error: {0}")]
    SessionError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
}

pub type GypsyResult<T> = Result<T, GypsyError>;

// Usage:
match result {
    Err(GypsyError::ConfigError(msg)) => { ... },
    Err(GypsyError::SessionError(msg)) => { ... },
    _ => { ... },
}