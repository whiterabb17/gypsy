use gypsy::agent_manager::{AgentEvent, AgentManager};
use gypsy::config::AppConfig;
use gypsy::ui::{ui, AppState, LogEntry};

use anyhow::{Context as _, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use reqwest;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use std::sync::Mutex;
use ringbuffer::RingBuffer;

static OLLAMA_CHILD: once_cell::sync::Lazy<Arc<Mutex<Option<tokio::process::Child>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(None)));

// --- Ollama Management ---

async fn ensure_ollama_ready(config: &AppConfig) -> Result<()> {
    if config.provider != "ollama" {
        return Ok(());
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let health_url = format!("{}/api/tags", config.ollama_base_url);

    // 1. Check Health (Auto-start)
    if client.get(&health_url).send().await.is_err() {
        tracing::info!("Ollama not running. Attempting to start...");
        start_ollama_daemon().await?;

        // Wait for startup (max 30s)
        for _ in 0..60 {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if client.get(&health_url).send().await.is_ok() {
                break;
            }
        }
    }

    // 2. Verify/Pull Models
    let ollama = ollama_rs::Ollama::new("http://localhost".to_string(), 11434);

    let local_models = ollama.list_local_models().await.unwrap_or_default();
    let model_names: Vec<String> = local_models.into_iter().map(|m| m.name).collect();

    if !model_names.contains(&config.model_name) {
        tracing::info!("Pulling LLM model: {}...", config.model_name);
        ollama.pull_model(config.model_name.clone(), false).await?;
    }

    if !model_names.contains(&config.embedding_model) {
        tracing::info!("Pulling embedding model: {}...", config.embedding_model);
        ollama
            .pull_model(config.embedding_model.clone(), false)
            .await?;
    }

    // 3. Pre-load Models (Force initial 40s load to happen now instead of crashing later)
    tracing::info!("Pre-loading LLM model: {}...", config.model_name);
    let _ = ollama.generate(ollama_rs::generation::completion::request::GenerationRequest::new(config.model_name.clone(), "".to_string())).await;
    
    tracing::info!("Pre-loading embedding model: {}...", config.embedding_model);
    let _ = ollama.generate(ollama_rs::generation::completion::request::GenerationRequest::new(config.embedding_model.clone(), "".to_string())).await;

    Ok(())
}

async fn start_ollama_daemon() -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let paths = [
            "/usr/local/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
            "/Applications/Ollama.app/Contents/MacOS/ollama",
        ];
        for path in paths {
            if std::path::Path::new(path).exists() {
                let mut cmd = tokio::process::Command::new(path);
                if path.contains("Resources") || path.contains("bin") {
                    cmd.arg("serve");
                }
                if let Ok(child) = cmd.spawn() {
                    if let Ok(mut guard) = OLLAMA_CHILD.lock() {
                        *guard = Some(child);
                    }
                    return Ok(());
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        let local_app_data = std::env::var("LOCALAPPDATA").unwrap_or_default();
        let path = format!("{}\\Ollama\\ollama.exe", local_app_data);
        if std::path::Path::new(&path).exists() {
            if let Ok(child) = tokio::process::Command::new(path).arg("serve").spawn() {
                if let Ok(mut guard) = OLLAMA_CHILD.lock() {
                    *guard = Some(child);
                }
            }
            return Ok(());
        }
    }

    // Generic Linux/PATH fallback
    if let Ok(child) = tokio::process::Command::new("ollama").arg("serve").spawn() {
        if let Ok(mut guard) = OLLAMA_CHILD.lock() {
            *guard = Some(child);
        }
    }

    Ok(())
}

// --- Provider Factory ---

struct ProviderStack {
    pub model: Arc<dyn mentalist::ModelProvider>,
    pub embeddings: Arc<dyn mem_core::EmbeddingProvider>,
    pub token_counter: Arc<dyn mem_core::TokenCounter>,
}

fn create_providers(config: &AppConfig) -> Result<ProviderStack> {
    let (model, embeddings, token_counter): (
        Arc<dyn mentalist::ModelProvider>,
        Arc<dyn mem_core::EmbeddingProvider>,
        Arc<dyn mem_core::TokenCounter>,
    ) = match config.provider.to_lowercase().as_str() {
        "anthropic" => {
            let key = config.anthropic_api_key.as_ref().context("ANTHROPIC_API_KEY missing")?.clone();
            let model = config.model_name.clone();
            let provider = Arc::new(mem_core::AnthropicProvider::new(key, model));
            (
                provider.clone() as Arc<dyn mentalist::ModelProvider>,
                provider.clone() as Arc<dyn mem_core::EmbeddingProvider>,
                provider as Arc<dyn mem_core::TokenCounter>,
            )
        }
        "openai" => {
            let key = config.openai_api_key.as_ref().context("OPENAI_API_KEY missing")?.clone();
            let model = config.model_name.clone();
            let provider = Arc::new(mem_core::OpenAiProvider::new(key, model));
            (
                provider.clone() as Arc<dyn mentalist::ModelProvider>,
                provider.clone() as Arc<dyn mem_core::EmbeddingProvider>,
                provider as Arc<dyn mem_core::TokenCounter>,
            )
        }
        "gemini" => {
            let key = config.gemini_api_key.as_ref().context("GEMINI_API_KEY missing")?.clone();
            let model = config.model_name.clone();
            let provider = Arc::new(mem_core::GeminiProvider::new(key, model));
            (
                provider.clone() as Arc<dyn mentalist::ModelProvider>,
                provider.clone() as Arc<dyn mem_core::EmbeddingProvider>,
                provider as Arc<dyn mem_core::TokenCounter>,
            )
        }
        _ => {
            let m = config.model_name.clone();
            let e = config.embedding_model.clone();
            let provider = Arc::new(mem_core::OllamaProvider::new(m, e));
            (
                provider.clone() as Arc<dyn mentalist::ModelProvider>,
                provider.clone() as Arc<dyn mem_core::EmbeddingProvider>,
                provider as Arc<dyn mem_core::TokenCounter>,
            )
        }
    };
    Ok(ProviderStack { model, embeddings, token_counter })
}

pub struct MockLlmClient;
#[async_trait::async_trait]
impl mem_core::LlmClient for MockLlmClient {
    async fn completion(&self, _prompt: &str) -> Result<String> {
        Ok("Mock completion".to_string())
    }
}

// --- Log Layer ---

struct LogVisitor {
    message: String,
}

impl tracing::field::Visit for LogVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{:?}", value);
        }
    }
}

struct UiLogLayer {
    tx: mpsc::Sender<LogEntry>,
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for UiLogLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut visitor = LogVisitor {
            message: String::new(),
        };
        event.record(&mut visitor);
        if !visitor.message.is_empty() {
            let metadata = event.metadata();
            let level = metadata.level();
            let target = metadata.target();
            let msg = format!("{}: {}", target, visitor.message);

            let entry = match *level {
                tracing::Level::ERROR => LogEntry::Error(msg),
                tracing::Level::WARN => LogEntry::Warn(msg),
                tracing::Level::INFO => LogEntry::Info(msg),
                tracing::Level::DEBUG => LogEntry::Debug(msg),
                tracing::Level::TRACE => LogEntry::Trace(msg),
            };
            if let Err(mpsc::error::TrySendError::Full(_)) = self.tx.try_send(entry) {
                // Drop log if channel full, but don't panic
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // 0. Setup Terminal Early (Pre-Async setup)
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // 1. Setup Logging (Redirect tracing to MPSC)
    let (log_tx, mut log_rx) = mpsc::channel::<LogEntry>(5000);
    tracing_subscriber::registry()
        .with(UiLogLayer { tx: log_tx })
        .init();

    // 2. Load Config & Verify Provider (Ollama auto-start)
    let config = AppConfig::from_env();

    if let Err(e) = ensure_ollama_ready(&config).await {
        tracing::error!("Ollama init failure: {}. Continuing...", e);
    }

    let providers = create_providers(&config)?;

    // 3. Setup Agent & Communication
    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(1000);

    let mut manager = AgentManager::new(
        event_tx.clone(),
        providers.model.clone(),
        providers.embeddings,
        providers.token_counter,
        config,
    ).await?;

    let mut state = AppState::new();
    state.available_commands = manager.get_available_commands().await?;
    state.log.push(LogEntry::Info(
        "Gypsy Shell Initialized.🔮 Ready for input.".into(),
    ));

    // 4. Input & Agent Loop
    let (input_tx, mut input_rx) = mpsc::channel::<String>(100);

    let event_tx_clone = event_tx.clone();
    tokio::spawn(async move {
        while let Some(input) = input_rx.recv().await {
            if let Err(e) = manager.run_step(input).await {
                let _ = event_tx_clone.send(AgentEvent::Error(format!("Agent Error: {}", e)));
            }
        }
    });

    'main_loop: loop {
        // Separate counts for scrolling
        let (chat_count, system_count) = state.log.iter().fold((0usize, 0usize), |(c, s), e| {
            match e {
                LogEntry::User(_) | LogEntry::Gypsy(_) | LogEntry::Error(_) => (c + 1, s),
                LogEntry::Info(_) | LogEntry::Warn(_) => (c, s + 1),
                LogEntry::Debug(_) | LogEntry::Trace(_) if state.show_debug => (c, s + 1),
                _ => (c, s),
            }
        });

        // Drain Background Logs
        let mut new_logs = false;
        while let Ok(log_msg) = log_rx.try_recv() {
            state.log.push(log_msg);
            new_logs = true;
        }

        if new_logs {
            // Auto-scroll logic for chat (Interaction Log)
            if state.log_scroll >= chat_count.saturating_sub(5) as u16 {
                state.log_scroll = chat_count as u16;
            }
            
            // Auto-scroll logic for system log (Under-the-Hood)
            if state.system_log_scroll >= system_count.saturating_sub(5) as u16 {
                state.system_log_scroll = system_count as u16;
            }
        }

        terminal.draw(|f| ui(f, &state))?;

        if event::poll(Duration::from_millis(10))? {
            if let Event::Key(key) = event::read()? {
                // Fix for Windows duplication: only handle Press events
                if key.kind != event::KeyEventKind::Press {
                    continue 'main_loop;
                }

                match key.code {
                    KeyCode::Esc => break,
                    KeyCode::F(1) => {
                        state.show_debug = !state.show_debug;
                    }
                    KeyCode::PageUp => {
                        if key.modifiers.contains(event::KeyModifiers::SHIFT) {
                            state.system_log_scroll = state.system_log_scroll.saturating_sub(5);
                        } else {
                            state.log_scroll = state.log_scroll.saturating_sub(5);
                        }
                    }
                    KeyCode::PageDown => {
                        if key.modifiers.contains(event::KeyModifiers::SHIFT) {
                            state.system_log_scroll = state.system_log_scroll.saturating_add(5);
                        } else {
                            state.log_scroll = state.log_scroll.saturating_add(5);
                        }
                    }
                    KeyCode::Tab => {
                        if !state.autocomplete_suggestions.is_empty() {
                            state.input_buffer = state.autocomplete_suggestions[0].clone();
                            state.autocomplete_suggestions.clear();
                        }
                    }
                    KeyCode::Enter => {
                        let input: String = state.input_buffer.drain(..).collect();
                        state.autocomplete_suggestions.clear();
                        if !input.trim().is_empty() {
                            state.log.push(LogEntry::User(input.clone()));
                            state.log_scroll = chat_count as u16; // Scroll to bottom
                            state.status = "Thinking...".to_string();
                            state.is_thinking = true;
                            if let Err(_) = input_tx.try_send(input) {
                                state.log.push(LogEntry::Error("Input channel full".into()));
                            }
                        }
                    }
                    KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                        break;
                    }
                    KeyCode::Char(c) => {
                        if state.input_buffer.len() < 4096 {
                            state.input_buffer.push(c);
                        }
                        if state.input_buffer.starts_with('/') {
                            state.autocomplete_suggestions = state
                                .available_commands
                                .iter()
                                .filter(|cmd| cmd.starts_with(&state.input_buffer))
                                .cloned()
                                .collect();
                        } else {
                            state.autocomplete_suggestions.clear();
                        }
                    }
                    KeyCode::Backspace => {
                        state.input_buffer.pop();
                        if state.input_buffer.starts_with('/') && !state.input_buffer.is_empty() {
                            state.autocomplete_suggestions = state
                                .available_commands
                                .iter()
                                .filter(|cmd| cmd.starts_with(&state.input_buffer))
                                .cloned()
                                .collect();
                        } else {
                            state.autocomplete_suggestions.clear();
                        }
                    }
                    _ => {}
                }
            }
        }

        // Handle Agent Events
        while let Ok(event) = event_rx.try_recv() {
            match event {
                AgentEvent::Quit => {
                    break 'main_loop;
                }
                AgentEvent::Status(s) => {
                    if s == "Idle" {
                        state.is_thinking = false;
                    }
                    state.status = s;
                }
                AgentEvent::TextChunk(c) => {
                        if let Some(last_entry) = state.log.iter_mut().last() {
                            if let LogEntry::Gypsy(ref mut msg) = last_entry {
                                msg.push_str(&c);
                            } else {
                                state.log.push(LogEntry::Gypsy(c));
                            }
                        } else {
                            state.log.push(LogEntry::Gypsy(c));
                        }
                    }
                AgentEvent::MetricUpdate {
                    tokens,
                    context_size,
                    step,
                } => {
                    state.tokens += tokens;
                    state.context_size = context_size;
                    state.current_step = step;
                }
                AgentEvent::Error(e) => {
                    state.log.push(LogEntry::Error(e));
                    state.status = "Error State".to_string();
                    state.is_thinking = false;
                }
            }
        }

        if state.is_thinking {
            state.counter += 1;
        }

        tokio::task::yield_now().await;
    }

    // 4. Cleanup
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    // Set cleanup logic
    if let Ok(mut guard) = OLLAMA_CHILD.lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
        }
    }

    Ok(())
}
