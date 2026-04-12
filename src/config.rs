use mem_core::MindPalaceConfig;
use secrecy::Secret;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    // LLM & Provider
    pub provider: String,
    pub model_name: String,
    pub embedding_model: String,
    pub ollama_base_url: String,
    pub anthropic_api_key: Option<Secret<String>>,
    pub openai_api_key: Option<Secret<String>>,
    pub gemini_api_key: Option<Secret<String>>,
    pub embedding_dimension: usize,

    // MindPalace Core
    pub similarity_threshold: f32,
    pub compression_ratio: f32,
    pub max_context_items: usize,
    pub base_ttl_seconds: u64,
    pub summary_interval: usize,
    pub max_tokens_per_dream: usize,

    // Resilience
    pub failure_threshold: u32,

    // Sandbox
    pub sandbox_mode: String,
    pub docker_image: String,
    pub ram_limit_mb: u64,
    pub cpu_limit_percent: u64,
    pub wasm_module_path: Option<String>,
    #[serde(default)]
    pub wasm_env_vars: HashMap<String, String>,

    // Storage & Session
    pub storage_path: String,
    pub sessions_path: String,
    pub mcp_root_path: String,
    pub vault_path: Option<String>,
    #[serde(default)]
    pub session_id: String,

    // Tools
    #[serde(default)]
    pub mcp_servers: HashMap<String, String>,
    #[serde(default)]
    pub mcp_filesystem_paths: Vec<String>,
    pub firecrawl_api_key: Option<Secret<String>>,
    pub skills_path: String,
    pub log_level: String,
    pub model_context_window: usize,
    pub personality_instructions: Option<String>,
    pub system_prompt: Option<String>,
    pub fallback_mode: String,
    pub enable_ddg_search: bool,
    pub mcp_initialize_timeout_seconds: u64,
    pub max_steps: usize,
}

impl AppConfig {
    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();

        let mut s = config::Config::builder()
            // Defaults
            .set_default("provider", "ollama")
            .unwrap()
            .set_default("model_name", "gemma4:26b")
            .unwrap()
            .set_default("embedding_model", "nomic-embed-text")
            .unwrap()
            .set_default("ollama_base_url", "http://127.0.0.1:11434")
            .unwrap()
            .set_default("embedding_dimension", 768)
            .unwrap()
            .set_default("similarity_threshold", 0.85)
            .unwrap()
            .set_default("compression_ratio", 0.6)
            .unwrap()
            .set_default("max_context_items", 100)
            .unwrap()
            .set_default("base_ttl_seconds", 3600)
            .unwrap()
            .set_default("summary_interval", 15)
            .unwrap()
            .set_default("max_tokens_per_dream", 50000)
            .unwrap()
            .set_default("failure_threshold", 3)
            .unwrap()
            .set_default("sandbox_mode", "local")
            .unwrap()
            .set_default("docker_image", "alpine:latest")
            .unwrap()
            .set_default("ram_limit_mb", 1024)
            .unwrap()
            .set_default("cpu_limit_percent", 50)
            .unwrap()
            .set_default("storage_path", ".agent/storage")
            .unwrap()
            .set_default("sessions_path", ".agent/sessions")
            .unwrap()
            .set_default("mcp_root_path", ".agent/mcp")
            .unwrap()
            .set_default("skills_path", ".agent/skills")
            .unwrap()
            .set_default("log_level", "info")
            .unwrap()
            .set_default("model_context_window", 262144)
            .unwrap()
            .set_default("session_id", "")
            .unwrap()
            .set_default("fallback_mode", "automatic")
            .unwrap()
            .set_default("enable_ddg_search", true)
            .unwrap()
            .set_default("mcp_initialize_timeout_seconds", 60)
            .unwrap()
            .set_default("max_steps", 10)
            .unwrap();

        // Environment overrides
        s = s.add_source(config::Environment::default().separator("__"));

        let mut config: AppConfig = s.build().unwrap().try_deserialize().unwrap();

        // Post-processing for dynamic defaults
        if config.session_id.is_empty() {
            let now = chrono::Utc::now();
            config.session_id = format!("gypsy_{}", now.format("%Y%m%d_%H%M%S"));
        }

        // Debug assertions override
        #[cfg(debug_assertions)]
        {
            if std::env::var("STORAGE_PATH").is_err() {
                config.storage_path = "./memory".to_string();
            }
        }

        // Manual collection of prefixed env vars not handled by config-rs easily
        for (k, v) in std::env::vars() {
            if k.starts_with("WASM_ENV_") {
                config
                    .wasm_env_vars
                    .insert(k.trim_start_matches("WASM_ENV_").to_string(), v);
            } else if k.starts_with("MCP_SERVER_") {
                config
                    .mcp_servers
                    .insert(k.trim_start_matches("MCP_SERVER_").to_lowercase(), v);
            }
        }

        if let Ok(fs_paths) = std::env::var("MCP_FS_PATHS") {
            config.mcp_filesystem_paths =
                fs_paths.split(',').map(|s| s.trim().to_string()).collect();
        }

        config
    }

    pub fn to_mindpalace_config(&self) -> MindPalaceConfig {
        MindPalaceConfig {
            default_model: self.model_name.clone(),
            similarity_threshold: self.similarity_threshold,
            compression_ratio: self.compression_ratio,
            max_context_items: self.max_context_items,
            base_ttl_seconds: self.base_ttl_seconds,
            summary_interval: self.summary_interval,
            max_tokens_per_dream: self.max_tokens_per_dream,
            ..Default::default()
        }
    }

    pub fn to_security_config(&self) -> mentalist::config::SecurityConfig {
        mentalist::config::SecurityConfig {
            max_memory_mb: self.ram_limit_mb,
            enforce_sandboxing: self.sandbox_mode != "local",
            ..Default::default()
        }
    }

    pub fn to_agent_config(&self) -> mentalist::config::AgentConfig {
        mentalist::config::AgentConfig {
            max_context_items: self.max_context_items,
            timeout_seconds: 300, // Default or add to AppConfig
            ..Default::default()
        }
    }
}
