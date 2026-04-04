use serde::Deserialize;
use mem_core::MindPalaceConfig;

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    // LLM & Provider
    pub provider: String,
    pub model_name: String,
    pub embedding_model: String,
    pub ollama_base_url: String,
    pub anthropic_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub gemini_api_key: Option<String>,

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
    pub wasm_env_vars: std::collections::HashMap<String, String>,

    // Storage & Session
    pub storage_path: String,
    pub vault_path: Option<String>,
    pub session_id: String,
}

impl AppConfig {
    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();
        
        let mut storage_path = std::env::var("STORAGE_PATH").unwrap_or_else(|_| ".agent/storage".to_string());
        
        #[cfg(debug_assertions)]
        {
            if std::env::var("STORAGE_PATH").is_err() {
                storage_path = "./memory".to_string();
            }
        }

        let mut wasm_env_vars = std::collections::HashMap::new();
        for (k, v) in std::env::vars() {
            if k.starts_with("WASM_ENV_") {
                wasm_env_vars.insert(k.trim_start_matches("WASM_ENV_").to_string(), v);
            }
        }

        Self {
            provider: std::env::var("PROVIDER").unwrap_or_else(|_| "ollama".to_string()),
            model_name: std::env::var("MODEL_NAME").unwrap_or_else(|_| "llama3".to_string()),
            embedding_model: std::env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string()),
            ollama_base_url: std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string()),
            anthropic_api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            gemini_api_key: std::env::var("GEMINI_API_KEY").ok(),

            similarity_threshold: std::env::var("SIMILARITY_THRESHOLD").ok().and_then(|v| v.parse().ok()).unwrap_or(0.85),
            compression_ratio: std::env::var("COMPRESSION_RATIO").ok().and_then(|v| v.parse().ok()).unwrap_or(0.6),
            max_context_items: std::env::var("MAX_CONTEXT_ITEMS").ok().and_then(|v| v.parse().ok()).unwrap_or(100),
            base_ttl_seconds: std::env::var("BASE_TTL_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3600),
            summary_interval: std::env::var("SUMMARY_INTERVAL").ok().and_then(|v| v.parse().ok()).unwrap_or(15),
            max_tokens_per_dream: std::env::var("MAX_TOKENS_PER_DREAM").ok().and_then(|v| v.parse().ok()).unwrap_or(50000),

            failure_threshold: std::env::var("FAILURE_THRESHOLD").ok().and_then(|v| v.parse().ok()).unwrap_or(3),

            sandbox_mode: std::env::var("SANDBOX_MODE").unwrap_or_else(|_| "local".to_string()),
            docker_image: std::env::var("DOCKER_IMAGE").unwrap_or_else(|_| "alpine:latest".to_string()),
            ram_limit_mb: std::env::var("RAM_LIMIT_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(4096), // 4GB default
            cpu_limit_percent: std::env::var("CPU_LIMIT_PERCENT").ok().and_then(|v| v.parse().ok()).unwrap_or(50),
            wasm_module_path: std::env::var("WASM_MODULE_PATH").ok(),
            wasm_env_vars,

            storage_path,
            vault_path: std::env::var("VAULT_PATH").ok(),
            session_id: std::env::var("SESSION_ID").unwrap_or_else(|_| "gypsy_dev_001".to_string()),
        }
    }

    pub fn to_mindpalace_config(&self) -> MindPalaceConfig {
        let mut config = MindPalaceConfig::default();
        config.default_model = self.model_name.clone();
        config.similarity_threshold = self.similarity_threshold;
        config.compression_ratio = self.compression_ratio;
        config.max_context_items = self.max_context_items;
        config.base_ttl_seconds = self.base_ttl_seconds;
        config.summary_interval = self.summary_interval;
        config.max_tokens_per_dream = self.max_tokens_per_dream;
        config
    }
}
