use std::process::Stdio;
use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::Result;
use crate::config::AppConfig;
use std::time::Duration;
use reqwest;
use crate::error::{GypsyError, GypsyResult};

pub struct ServiceManager {
    ollama_child: Arc<Mutex<Option<tokio::process::Child>>>,
}

impl ServiceManager {
    pub fn new() -> Self {
        Self {
            ollama_child: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn ensure_ollama_ready(&self, config: &AppConfig) -> GypsyResult<()> {
        if config.provider != "ollama" {
            return Ok(());
        }

        match self.check_health(config).await {
            Ok(true) => {
                tracing::info!("Ollama is already running and healthy.");
            }
            _ => {
                tracing::info!("Ollama not running or unhealthy. Attempting to start...");
                self.start_ollama_daemon().await?;

                // Wait for startup (max 30s)
                let client = reqwest::Client::builder()
                    .timeout(Duration::from_secs(2))
                    .build()?;
                let health_url = format!("{}/api/tags", config.ollama_base_url);
                
                let mut ready = false;
                for _ in 0..60 {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    if client.get(&health_url).send().await.is_ok() {
                        ready = true;
                        break;
                    }
                }
                if !ready {
                    return Err(GypsyError::AgentError("Ollama failed to start within 30 seconds".into()));
                }
            }
        }

        self.ensure_models_pulled(config).await
    }

    pub async fn check_health(&self, config: &AppConfig) -> GypsyResult<bool> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()?;
        let health_url = format!("{}/api/tags", config.ollama_base_url);
        
        Ok(client.get(&health_url).send().await.is_ok())
    }

    pub async fn ensure_models_pulled(&self, config: &AppConfig) -> GypsyResult<()> {
        let url = config.ollama_base_url.parse::<reqwest::Url>().map_err(|e| GypsyError::ConfigError(e.to_string()))?;
        let scheme = url.scheme();
        let mut host_val = url.host_str().unwrap_or("127.0.0.1").to_string();
        if host_val == "localhost" {
            host_val = "127.0.0.1".to_string();
        }
        let base = format!("{}://{}", scheme, host_val);
        let port = url.port().unwrap_or(11434);
        
        let ollama = ollama_rs::Ollama::new(base, port);
        
        // Capture specific errors during model inventory check to avoid silent failures
        let local_models = ollama.list_local_models().await?;
        let model_names: Vec<String> = local_models.into_iter().map(|m| m.name).collect();

        if !model_names.contains(&config.model_name) && !model_names.iter().any(|n| n.starts_with(&config.model_name)) {
            tracing::info!("Pulling LLM model: {}...", config.model_name);
            ollama.pull_model(config.model_name.clone(), false).await?;
        }

        if !model_names.contains(&config.embedding_model) && !model_names.iter().any(|n| n.starts_with(&config.embedding_model)) {
            tracing::info!("Pulling embedding model: {}...", config.embedding_model);
            ollama.pull_model(config.embedding_model.clone(), false).await?;
        }

        Ok(())
    }

    async fn start_ollama_daemon(&self) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            // Try to kill orphans that might be leaking logs into our terminal
            let _ = std::process::Command::new("taskkill")
                .args(["/F", "/IM", "ollama.exe"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();

            let local_app_data = std::env::var("LOCALAPPDATA").unwrap_or_default();
            let path = format!("{}\\Ollama\\ollama.exe", local_app_data);
            if std::path::Path::new(&path).exists() {
                let child = tokio::process::Command::new(path)
                    .arg("serve")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()?;
                *self.ollama_child.lock().await = Some(child);
                return Ok(());
            }
        }

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
                    cmd.stdout(Stdio::null());
                    cmd.stderr(Stdio::null());

                    if let Ok(child) = cmd.spawn() {
                        *self.ollama_child.lock().await = Some(child);
                        return Ok(());
                    }
                }
            }
        }

        // Generic Linux / Path fallback
        let child = tokio::process::Command::new("ollama")
            .arg("serve")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        *self.ollama_child.lock().await = Some(child);
        Ok(())
    }

    pub async fn restart_service(&self, config: &AppConfig) -> GypsyResult<()> {
        tracing::info!("Restarting Ollama service...");
        self.shutdown().await;
        tokio::time::sleep(Duration::from_secs(1)).await;
        self.ensure_ollama_ready(config).await
    }

    pub async fn shutdown(&self) {
        if let Some(mut child) = self.ollama_child.lock().await.take() {
            let _ = child.kill().await;
        }
    }
}
