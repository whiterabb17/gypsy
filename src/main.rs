use gypsy::agent_manager::{AgentEvent, AgentManager};
use gypsy::config::AppConfig;
use gypsy::prefs::PrefsManager;
use gypsy::service::ServiceManager;
use gypsy::ui::{ui, AppState, LogEntry};

use anyhow::{Context as _, Result};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use ringbuffer::RingBuffer;
use secrecy::ExposeSecret;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

struct ProviderStack {
    pub model: Arc<dyn mentalist::ModelProvider>,
    pub embeddings: Arc<dyn mem_core::EmbeddingProvider>,
    pub token_counter: Arc<dyn mem_core::TokenCounter>,
    pub context_window: usize,
    pub embedding_dimension: usize,
}


async fn create_providers(
    config: &AppConfig,
    _event_tx: mpsc::Sender<AgentEvent>,
) -> Result<ProviderStack> {
    let mut embedding_dimension = config.embedding_dimension;
    let mut context_window = config.model_context_window;

    let (primary_model, embeddings, token_counter): (
        Arc<dyn mentalist::ModelProvider>,
        Arc<dyn mem_core::EmbeddingProvider>,
        Arc<dyn mem_core::TokenCounter>,
    ) = match config.provider.to_lowercase().as_str() {
        "anthropic" => {
            let key = config
                .anthropic_api_key
                .as_ref()
                .context("ANTHROPIC_API_KEY missing")?
                .expose_secret()
                .clone();
            let model_name = config.model_name.clone();
            let provider = Arc::new(mem_core::AnthropicProvider::new(key, model_name));
            
            // Minimal config for cloud: auto-resolve
            embedding_dimension = 1536; 
            context_window = 200000;

            (
                provider.clone() as Arc<dyn mentalist::ModelProvider>,
                provider.clone() as Arc<dyn mem_core::EmbeddingProvider>,
                provider as Arc<dyn mem_core::TokenCounter>,
            )
        }
        "openai" => {
            let key = config
                .openai_api_key
                .as_ref()
                .context("OPENAI_API_KEY missing")?
                .expose_secret()
                .clone();
            let model_name = config.model_name.clone();
            let provider = Arc::new(mem_core::OpenAiProvider::new(key, model_name));
            
            // Minimal config for cloud: auto-resolve
            embedding_dimension = 1536;
            context_window = 128000;

            (
                provider.clone() as Arc<dyn mentalist::ModelProvider>,
                provider.clone() as Arc<dyn mem_core::EmbeddingProvider>,
                provider as Arc<dyn mem_core::TokenCounter>,
            )
        }
        "gemini" => {
            let key = config
                .gemini_api_key
                .as_ref()
                .context("GEMINI_API_KEY missing")?
                .expose_secret()
                .clone();
            let model_name = config.model_name.clone();
            let provider = Arc::new(mem_core::GeminiProvider::new(key, model_name));
            
            // Minimal config for cloud: auto-resolve
            embedding_dimension = 768; // Gemini 1.5 defaults
            context_window = 1000000;

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
            let provider = Arc::new(mem_core::OllamaProvider::new(
                config.ollama_base_url.clone(),
                m,
                e,
                ctx,
            ));
            
            // Ollama uses configs from .env (already initialized defaults above)

            (
                provider.clone() as Arc<dyn mentalist::ModelProvider>,
                provider.clone() as Arc<dyn mem_core::EmbeddingProvider>,
                provider as Arc<dyn mem_core::TokenCounter>,
            )
        }
    };

    // Try to discover more accurate context window if provider supports it
    let metadata = primary_model.discover_metadata().await;
    if metadata.context_window > 2048 {
        context_window = metadata.context_window;
    }

    Ok(ProviderStack {
        model: primary_model,
        embeddings,
        token_counter,
        context_window,
        embedding_dimension,
    })
}


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
// WASM engine initialization handled by Mentalist if needed.

    let config = AppConfig::from_env();

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

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, event::EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let service_manager = Arc::new(ServiceManager::new());
    if let Err(e) = service_manager.ensure_ollama_ready(&config).await {
        tracing::error!(
            "Ollama initialization failed: {}. Some features may be unavailable.",
            e
        );
    }

    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(1000);
    let (input_tx, mut input_rx) = mpsc::channel::<String>(100);

    let providers = create_providers(&config, event_tx.clone()).await?;
    
    let pref_dir = std::path::PathBuf::from(".agent");
    let prefs = Arc::new(tokio::sync::Mutex::new(PrefsManager::new(&pref_dir)));

    let mut manager = AgentManager::new(
        event_tx.clone(),
        providers.model.clone(),
        providers.embeddings,
        providers.token_counter,
        prefs,
        config.clone(),
        Some(providers.embedding_dimension),
        Some(providers.context_window),
    )
    .await?;

    let mut state = AppState::new();
    state.max_tokens = providers.context_window;

    state.available_commands = manager.get_available_commands().await?;
    state.log.push(LogEntry::Gypsy(
        "Gypsy agent is ready. Session initialized and secured. 🔮".into(),
    ));

    let event_tx_clone = event_tx.clone();
    tokio::spawn(async move {
        while let Some(input) = input_rx.recv().await {
            if let Err(e) = manager.run_step(input).await {
                let _ = event_tx_clone
                    .send(AgentEvent::Error(format!("Agent Error: {}", e)))
                    .await;
            }
        }
    });

    if config.provider == "ollama" {
        let service_manager_bg = service_manager.clone();
        let event_tx_bg = event_tx.clone();
        let config_bg = config.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            let mut failures = 0;
            loop {
                interval.tick().await;
                match service_manager_bg.check_health(&config_bg).await {
                    Ok(true) => {
                        failures = 0;
                    }
                    _ => {
                        failures += 1;
                        let _ = event_tx_bg
                            .send(AgentEvent::Status(format!(
                                "Ollama Offline (Failures: {})",
                                failures
                            )))
                            .await;
                        if failures >= 2 {
                            let _ = event_tx_bg
                                .send(AgentEvent::Status(
                                    "Attempting Ollama Auto-Restart...".into(),
                                ))
                                .await;
                            if let Err(e) = service_manager_bg.restart_service(&config_bg).await {
                                let _ = event_tx_bg
                                    .send(AgentEvent::Error(format!(
                                        "Ollama restart failed: {}",
                                        e
                                    )))
                                    .await;
                            } else {
                                failures = 0;
                                let _ = event_tx_bg
                                    .send(AgentEvent::Status("Ollama Recovered".into()))
                                    .await;
                            }
                        }
                    }
                }
            }
        });
    }

    let mut last_tick = std::time::Instant::now();
    let tick_rate = Duration::from_millis(50);

    'main_loop: loop {
        terminal.draw(|f| ui(f, &mut state))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::from_secs(0));

        if event::poll(timeout)? {
            match event::read()? {
                Event::Key(key) => {
                    if key.kind != event::KeyEventKind::Press {
                        continue;
                    }
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
                                state.log.push(LogEntry::Reasoning {
                                    main_step: "Initializing...".to_string(),
                                    sub_steps: Vec::new(),
                                    collapsed: false,
                                });
                                state.follow_chat = true;
                                state.status = "Thinking...".into();
                                state.is_thinking = true;
                                state.last_input_tokens = 0;
                                state.last_output_tokens = 0;
                                if let Err(e) = input_tx.try_send(input) {
                                    state
                                        .log
                                        .push(LogEntry::Error("Queue full, command dropped.".into()));
                                    tracing::warn!("Agent queue full: {:?}", e);
                                }
                            }
                        }
                        KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                            break 'main_loop
                        }
                        KeyCode::Char('y') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                            // Find the last agent response
                            if let Some(LogEntry::Gypsy(msg)) = state.log.iter().rev().find(|m| matches!(m, LogEntry::Gypsy(_))) {
                                match arboard::Clipboard::new() {
                                    Ok(mut clipboard) => {
                                        if let Err(e) = clipboard.set_text(msg.clone()) {
                                            state.log.push(LogEntry::Error(format!("Clipboard error: {}", e)));
                                        } else {
                                            state.log.push(LogEntry::Info("Gypsy response copied to clipboard.".into()));
                                        }
                                    }
                                    Err(e) => {
                                        state.log.push(LogEntry::Error(format!("Failed to init clipboard: {}", e)));
                                    }
                                }
                            }
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
                Event::Mouse(mouse) => {
                    if mouse.kind == event::MouseEventKind::Down(event::MouseButton::Left) {
                        for (idx, rect) in &state.reasoning_rects {
                            if mouse.column >= rect.x && mouse.column < rect.x + rect.width
                                && mouse.row >= rect.y && mouse.row < rect.y + rect.height
                            {
                                if let Some(LogEntry::Reasoning { collapsed, .. }) = state.log.get_mut(*idx) {
                                    *collapsed = !*collapsed;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        while let Ok(log_msg) = log_rx.try_recv() {
            state.log.push(log_msg);
        }
        while let Ok(event) = event_rx.try_recv() {
            match event {
                AgentEvent::Quit => break 'main_loop,
                AgentEvent::Status(s) => {
                    let old_thinking = state.is_thinking;
                    state.status = s.clone();
                    state.is_thinking = !["idle", "ready", "goal completed", "error"].iter().any(|&sub| state.status.to_lowercase().contains(sub));
                    state.pending_plan = None;

                    if state.is_thinking {
                        // Update reasoning log
                        if let Some(LogEntry::Reasoning { main_step, sub_steps, .. }) = state.log.iter_mut().rev().find(|m| matches!(m, LogEntry::Reasoning {..})) {
                            *main_step = s;
                            sub_steps.push(main_step.clone());
                        }
                    } else if old_thinking {
                        // Collapse on completion
                        if let Some(LogEntry::Reasoning { collapsed, .. }) = state.log.iter_mut().rev().find(|m| matches!(m, LogEntry::Reasoning {..})) {
                            *collapsed = true;
                        }
                    }
                }
                AgentEvent::TextChunk(c) => {
                    state.is_thinking = false;
                    state.pending_plan = None;
                    if let Some(LogEntry::Gypsy(ref mut msg)) = state.log.iter_mut().last() {
                        msg.push_str(&c);
                    } else {
                        state.log.push(LogEntry::Gypsy(c));
                    }
                }
                AgentEvent::MetricUpdate {
                    tokens,
                    input_tokens,
                    output_tokens,
                    context_size,
                    latency_ms,
                    step,
                    tool_name,
                } => {
                    // input_tokens and output_tokens are cumulative FOR THIS TURN.
                    // We calculate the delta and add it to the TOTAL session tokens.
                    let input_delta = input_tokens.saturating_sub(state.last_input_tokens);
                    let output_delta = output_tokens.saturating_sub(state.last_output_tokens);
                    
                    state.total_input_tokens += input_delta;
                    state.total_output_tokens += output_delta;
                    
                    state.last_input_tokens = input_tokens;
                    state.last_output_tokens = output_tokens;
                    
                    state.tokens = tokens; // Still used for turn-level display if needed
                    state.context_size = context_size;
                    state.current_step = step;

                    if latency_ms > 0 {
                        state.rolling_latency.push(latency_ms);
                        if state.rolling_latency.len() > 5 {
                            state.rolling_latency.remove(0);
                        }
                        state.llm_latency_ms = state.rolling_latency.iter().sum::<u128>()
                            / state.rolling_latency.len() as u128;
                    }

                    if let Some(tn) = tool_name {
                        state.last_tool_name = tn;
                        state.tool_calls_total += 1;
                    }
                }
                AgentEvent::Progress(p) => {
                    state.progress = Some(p);
                }
                AgentEvent::PhaseProgress(p) => {
                    state.phase_progress = Some(p);
                }
                AgentEvent::ToolResult { name, success } => {
                    state.tool_status.insert(name, success);
                }
                AgentEvent::AwaitingApproval(plan) => {
                    state.status = "Awaiting Approval".into();
                    state.is_thinking = false;
                    state.pending_plan = Some(plan.clone());
                    state.log.push(LogEntry::System(format!("Plan requires approval ({} tasks). Type /approve to proceed.", plan.tasks.len())));
                }
                AgentEvent::Error(e) => {
                    state.log.push(LogEntry::Error(e));
                    state.status = "Error".into();
                    state.is_thinking = false;
                    // Collapse on error
                    if let Some(LogEntry::Reasoning { collapsed, .. }) = state.log.iter_mut().rev().find(|m| matches!(m, LogEntry::Reasoning {..})) {
                        *collapsed = true;
                    }
                }
            }
        }

        if state.is_thinking {
            state.counter += 1;
        }

        if last_tick.elapsed() >= Duration::from_secs(5) {
            state.context_history.push(state.context_size as u64);
            if state.context_history.len() > 50 {
                state.context_history.remove(0);
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = std::time::Instant::now();
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, event::DisableMouseCapture)?;
    terminal.show_cursor()?;

    tracing::info!("Shutting down Gypsy services...");
    service_manager.shutdown().await;

    Ok(())
}
