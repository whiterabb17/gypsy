use gypsy::config::AppConfig;
use gypsy::agent_manager::{AgentEvent, AgentManager};
use gypsy::ui::{AppState, ui};

use anyhow::{Result, Context as _};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::time::Duration;
use std::sync::Arc;
use tokio::sync::mpsc;
use reqwest;

// --- Ollama Management ---

async fn ensure_ollama_ready(config: &AppConfig) -> Result<()> {
    if config.provider != "ollama" {
        return Ok(());
    }

    let client = reqwest::Client::new();
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
    let ollama = ollama_rs::Ollama::new(
        "http://localhost".to_string(),
        11434
    );
    
    let local_models = ollama.list_local_models().await.unwrap_or_default();
    let model_names: Vec<String> = local_models.into_iter().map(|m| m.name).collect();

    if !model_names.contains(&config.model_name) {
        tracing::info!("Pulling LLM model: {}...", config.model_name);
        ollama.pull_model(config.model_name.clone(), false).await?;
    }

    if !model_names.contains(&config.embedding_model) {
        tracing::info!("Pulling embedding model: {}...", config.embedding_model);
        ollama.pull_model(config.embedding_model.clone(), false).await?;
    }

    Ok(())
}

async fn start_ollama_daemon() -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let paths = [
            "/usr/local/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
            "/Applications/Ollama.app/Contents/MacOS/ollama"
        ];
        for path in paths {
            if std::path::Path::new(path).exists() {
                // The MacOS/ollama path might be the app wrapper, so we try without args there or handle it
                let mut cmd = tokio::process::Command::new(path);
                if path.contains("Resources") || path.contains("bin") {
                    cmd.arg("serve");
                }
                if cmd.spawn().is_ok() {
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
            tokio::process::Command::new(path)
                .arg("serve")
                .spawn()?;
            return Ok(());
        }
    }

    // Generic Linux/PATH fallback
    tokio::process::Command::new("ollama")
        .arg("serve")
        .spawn()?;
    
    Ok(())
}

// --- Provider Factory ---

struct ProviderStack {
    pub model: Box<dyn mentalist::ModelProvider>,
    pub embeddings: Arc<dyn mem_core::EmbeddingProvider>,
    pub token_counter: Arc<dyn mem_core::TokenCounter>,
}

fn create_providers(config: &AppConfig) -> Result<ProviderStack> {
    match config.provider.to_lowercase().as_str() {
        "anthropic" => {
            let key = config.anthropic_api_key.as_ref().context("ANTHROPIC_API_KEY missing")?;
            let provider = Arc::new(mem_core::AnthropicProvider::new(key.clone(), config.model_name.clone()));
            Ok(ProviderStack {
                model: Box::new(mem_core::AnthropicProvider::new(key.clone(), config.model_name.clone())),
                embeddings: provider.clone(),
                token_counter: provider,
            })
        }
        "openai" => {
            let key = config.openai_api_key.as_ref().context("OPENAI_API_KEY missing")?;
            let provider = Arc::new(mem_core::OpenAiProvider::new(key.clone(), config.model_name.clone()));
            Ok(ProviderStack {
                model: Box::new(mem_core::OpenAiProvider::new(key.clone(), config.model_name.clone())),
                embeddings: provider.clone(),
                token_counter: provider,
            })
        }
        "gemini" => {
            let key = config.gemini_api_key.as_ref().context("GEMINI_API_KEY missing")?;
            let provider = Arc::new(mem_core::GeminiProvider::new(key.clone(), config.model_name.clone()));
            Ok(ProviderStack {
                model: Box::new(mem_core::GeminiProvider::new(key.clone(), config.model_name.clone())),
                embeddings: provider.clone(),
                token_counter: provider,
            })
        }
        _ => {
            let provider = Arc::new(mem_core::OllamaProvider::new(config.model_name.clone(), config.embedding_model.clone()));
            Ok(ProviderStack {
                model: Box::new(mem_core::OllamaProvider::new(config.model_name.clone(), config.embedding_model.clone())),
                embeddings: provider.clone(),
                token_counter: provider,
            })
        }
    }
}

pub struct MockLlmClient;
#[async_trait::async_trait]
impl mem_core::LlmClient for MockLlmClient {
    async fn completion(&self, _prompt: &str) -> Result<String> {
        Ok("Mock completion".to_string())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Load Config & Verify Provider (Ollama auto-start)
    let config = AppConfig::from_env();
    tracing_subscriber::fmt::init();

    if let Err(e) = ensure_ollama_ready(&config).await {
        eprintln!("Failed to initialize Ollama: {}. Continuing with best effort...", e);
    }

    let providers = create_providers(&config)?;

    // 2. Setup Terminal
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // 3. Setup Agent & Communication
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AgentEvent>();
    
    let mut manager = AgentManager::new(
        event_tx.clone(),
        providers.model,
        providers.embeddings,
        providers.token_counter,
        config,
    )?;

    let mut state = AppState::new();
    state.log.push(">>> Gypsy Shell Initialized. Ready for input.".into());

    // 4. Input & Agent Loop
    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<String>();
    
    let event_tx_clone = event_tx.clone();
    tokio::spawn(async move {
        while let Some(input) = input_rx.recv().await {
            if let Err(e) = manager.run_step(input).await {
                let _ = event_tx_clone.send(AgentEvent::Error(format!("Agent Error: {}", e)));
            }
        }
    });

    'main_loop: loop {
        terminal.draw(|f| ui(f, &state))?;

        if event::poll(Duration::from_millis(10))? {
            if let Event::Key(key) = event::read()? {
                if let KeyCode::Esc = key.code {
                    break;
                }
                if let KeyCode::Char('c') = key.code {
                    if key.modifiers.contains(event::KeyModifiers::CONTROL) {
                        break;
                    }
                }
                match key.code {
                    KeyCode::Enter => {
                        let input: String = state.input_buffer.drain(..).collect();
                        if !input.trim().is_empty() {
                            state.log.push(format!(" User: {}", input));
                            state.status = "Thinking...".to_string();
                            state.is_thinking = true;
                            let _ = input_tx.send(input);
                        }
                    }
                    KeyCode::Char(c) => state.input_buffer.push(c),
                    KeyCode::Backspace => { state.input_buffer.pop(); }
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
                    if let Some(last) = state.log.last_mut() {
                        if !last.starts_with(" Gypsy: ") {
                            state.log.push(format!(" Gypsy: {}", c));
                        } else {
                            last.push_str(&c);
                        }
                    } else {
                        state.log.push(format!(" Gypsy: {}", c));
                    }
                    // Ensure thinking stays true until Idle or Message is received
                }
                AgentEvent::MetricUpdate { tokens, context_size, step } => {
                    state.tokens += tokens;
                    state.context_size = context_size;
                    state.current_step = step;
                }
                AgentEvent::Error(e) => {
                    state.log.push(format!(" Error: {}", e));
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

    Ok(())
}
