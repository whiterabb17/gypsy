use gypsy::agent_manager::{AgentEvent, AgentManager};
use gypsy::config::AppConfig;
use gypsy::ui::{ui, AppState, LogEntry};
use gypsy::service::ServiceManager;

use anyhow::{Context as _, Result};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use secrecy::ExposeSecret;
use ringbuffer::RingBuffer;

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
            let key = config.anthropic_api_key.as_ref().context("ANTHROPIC_API_KEY missing")?.expose_secret().clone();
            let model = config.model_name.clone();
            let provider = Arc::new(mem_core::AnthropicProvider::new(key, model));
            (
                provider.clone() as Arc<dyn mentalist::ModelProvider>,
                provider.clone() as Arc<dyn mem_core::EmbeddingProvider>,
                provider as Arc<dyn mem_core::TokenCounter>,
            )
        }
        "openai" => {
            let key = config.openai_api_key.as_ref().context("OPENAI_API_KEY missing")?.expose_secret().clone();
            let model = config.model_name.clone();
            let provider = Arc::new(mem_core::OpenAiProvider::new(key, model));
            (
                provider.clone() as Arc<dyn mentalist::ModelProvider>,
                provider.clone() as Arc<dyn mem_core::EmbeddingProvider>,
                provider as Arc<dyn mem_core::TokenCounter>,
            )
        }
        "gemini" => {
            let key = config.gemini_api_key.as_ref().context("GEMINI_API_KEY missing")?.expose_secret().clone();
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
            let ctx = Some(config.model_context_window as u32);
            let provider = Arc::new(mem_core::OllamaProvider::new(m, e, ctx));
            (
                provider.clone() as Arc<dyn mentalist::ModelProvider>,
                provider.clone() as Arc<dyn mem_core::EmbeddingProvider>,
                provider as Arc<dyn mem_core::TokenCounter>,
            )
        }
    };
    Ok(ProviderStack { model, embeddings, token_counter })
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
        let mut visitor = LogVisitor { message: String::new() };
        event.record(&mut visitor);
        if !visitor.message.is_empty() {
            let metadata = event.metadata();
            let msg = format!("{}: {}", metadata.target(), visitor.message);
            let entry = match *metadata.level() {
                tracing::Level::ERROR => LogEntry::Error(msg),
                tracing::Level::WARN => LogEntry::Warn(msg),
                tracing::Level::INFO => LogEntry::Info(msg),
                tracing::Level::DEBUG => LogEntry::Debug(msg),
                tracing::Level::TRACE => LogEntry::Trace(msg),
            };
            let _ = self.tx.try_send(entry);
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // 0. Initial Setup
    let config = AppConfig::from_env();
    
    // Setup Persistent File Logging
    let log_dir = ".agent/logs";
    let _ = std::fs::create_dir_all(log_dir);
    let file_appender = tracing_appender::rolling::daily(log_dir, "gypsy.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    let (log_tx, mut log_rx) = mpsc::channel::<LogEntry>(5000);
    let filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
        .parse_lossy(&config.log_level);

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(non_blocking))
        .with(UiLogLayer { tx: log_tx })
        .init();

    // 1. Setup Terminal
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // 2. Setup Services & Agents
    let service_manager = ServiceManager::new();
    if let Err(e) = service_manager.ensure_ollama_ready(&config).await {
        tracing::error!("Ollama initialization failed: {}. Some features may be unavailable.", e);
    }

    let providers = create_providers(&config)?;
    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(1000);
    let (input_tx, mut input_rx) = mpsc::channel::<String>(100);

    let mut manager = AgentManager::new(
        event_tx.clone(),
        providers.model.clone(),
        providers.embeddings,
        providers.token_counter,
        config.clone(),
    ).await?;

    let mut state = AppState::new();
    state.available_commands = manager.get_available_commands().await?;
    state.log.push(LogEntry::Gypsy("Gypsy agent is ready. Session initialized and secured. 🔮".into()));

    // 3. Agent Task
    let event_tx_clone = event_tx.clone();
    tokio::spawn(async move {
        while let Some(input) = input_rx.recv().await {
            if let Err(e) = manager.run_step(input).await {
                let _ = event_tx_clone.send(AgentEvent::Error(format!("Agent Error: {}", e))).await;
            }
        }
    });

    // 3.5 Ollama Health Task
    if config.provider == "ollama" {
        let service_manager_bg = Arc::new(service_manager);
        let event_tx_bg = event_tx.clone();
        let config_bg = config.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                match service_manager_bg.check_health(&config_bg).await {
                    Ok(true) => {
                        // Silent success
                    }
                    _ => {
                        let _ = event_tx_bg.send(AgentEvent::Error("Ollama connection lost! Status: Offline".into())).await;
                        let _ = event_tx_bg.send(AgentEvent::Status("Ollama Offline".into())).await;
                    }
                }
            }
        });
    } else {
        // Just to satisfy the shutdown call later if we wrap it in Arc
    }

    // 4. Main Event Loop
    let mut last_tick = std::time::Instant::now();
    let tick_rate = Duration::from_millis(50);

    'main_loop: loop {
        terminal.draw(|f| ui(f, &mut state))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::from_secs(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind != event::KeyEventKind::Press { continue; }
                match key.code {
                    KeyCode::Esc => break 'main_loop,
                    KeyCode::F(1) => state.show_debug = !state.show_debug,
                    KeyCode::PageUp => {
                        if key.modifiers.contains(event::KeyModifiers::SHIFT) {
                            state.system_log_scroll = state.system_log_scroll.saturating_sub(5);
                            state.follow_system = false;
                        } else {
                            state.log_scroll = state.log_scroll.saturating_sub(5);
                            state.follow_chat = false;
                        }
                    }
                    KeyCode::PageDown => {
                        if key.modifiers.contains(event::KeyModifiers::SHIFT) {
                            state.system_log_scroll = state.system_log_scroll.saturating_add(5);
                            state.follow_system = true;
                        } else {
                            state.log_scroll = state.log_scroll.saturating_add(5);
                            state.follow_chat = true;
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
                            state.follow_chat = true;
                            state.status = "Thinking...".into();
                            state.is_thinking = true;
                            if let Err(e) = input_tx.try_send(input) {
                                state.log.push(LogEntry::Error("Queue full, command dropped.".into()));
                                tracing::warn!("Agent queue full: {:?}", e);
                            }
                        }
                    }
                    KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => break 'main_loop,
                    KeyCode::Char(c) => {
                        if state.input_buffer.len() < 4096 { state.input_buffer.push(c); }
                        if state.input_buffer.starts_with('/') {
                            state.autocomplete_suggestions = state.available_commands.iter()
                                .filter(|cmd| cmd.starts_with(&state.input_buffer))
                                .cloned().collect();
                        } else { state.autocomplete_suggestions.clear(); }
                    }
                    KeyCode::Backspace => {
                        state.input_buffer.pop();
                        if state.input_buffer.starts_with('/') && !state.input_buffer.is_empty() {
                            state.autocomplete_suggestions = state.available_commands.iter()
                                .filter(|cmd| cmd.starts_with(&state.input_buffer))
                                .cloned().collect();
                        } else { state.autocomplete_suggestions.clear(); }
                    }
                    _ => {}
                }
            }
        }

        // Process Background Logs & Agent Events
        while let Ok(log_msg) = log_rx.try_recv() { state.log.push(log_msg); }
        while let Ok(event) = event_rx.try_recv() {
            match event {
                AgentEvent::Quit => break 'main_loop,
                AgentEvent::Status(s) => {
                    if s == "Idle" { state.is_thinking = false; }
                    state.status = s;
                }
                AgentEvent::TextChunk(c) => {
                    if let Some(LogEntry::Gypsy(ref mut msg)) = state.log.iter_mut().last() {
                        msg.push_str(&c);
                    } else { state.log.push(LogEntry::Gypsy(c)); }
                }
                AgentEvent::MetricUpdate { tokens, input_tokens, output_tokens, context_size, latency_ms, step, tool_name } => {
                    state.tokens = tokens;
                    state.total_input_tokens += input_tokens;
                    state.total_output_tokens += output_tokens;
                    state.context_size = context_size;
                    state.current_step = step;
                    if latency_ms > 0 { state.llm_latency_ms = latency_ms; }
                    if let Some(tn) = tool_name {
                        state.last_tool_name = tn;
                        state.tool_calls_total += 1;
                    }
                }
                AgentEvent::Error(e) => {
                    state.log.push(LogEntry::Error(e));
                    state.status = "Error".into();
                    state.is_thinking = false;
                }
            }
        }

        if state.is_thinking { state.counter += 1; }
        if last_tick.elapsed() >= tick_rate { last_tick = std::time::Instant::now(); }
    }

    // 5. Cleanup
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    
    // We can't easily call shutdown on service_manager context here if it was moved.
    // Let's ensure service_manager is an Arc from the start or just don't move it.
    
    Ok(())
}
