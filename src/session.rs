use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::sync::Arc;
use mentalist::DeepAgentState;
use mem_core::Context;

#[derive(Serialize, Deserialize)]
pub struct SessionFile {
    pub version: u32,
    pub state: DeepAgentState,
}

#[derive(Serialize, Deserialize, Default)]
pub struct SessionMetrics {
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
    pub tool_calls_total: u32,
    pub avg_latency_ms: u128,
    pub tool_success_rate: f32,
    pub session_duration_seconds: u64,
}

pub const CURRENT_SESSION_VERSION: u32 = 1;

#[derive(Clone)]
pub struct SessionManager {
    sessions_dir: PathBuf,
}

impl SessionManager {
    pub fn new(sessions_dir: impl Into<PathBuf>) -> Self {
        let dir = sessions_dir.into();
        let _ = std::fs::create_dir_all(&dir);
        Self { sessions_dir: dir }
    }

    pub fn get_session_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir.join(format!("session_{}.json", session_id))
    }

    pub fn load_session(&self, session_id: &str) -> Result<DeepAgentState> {
        let path = self.get_session_path(session_id);
        if !path.exists() {
            tracing::info!("No existing session found for {}, initializing new state.", session_id);
            return Ok(DeepAgentState {
                session_id: session_id.to_string(),
                context: Arc::new(Context { items: vec![] }),
                sandbox_root: std::env::current_dir()?,
            });
        }

        let data = std::fs::read_to_string(&path)?;
        
        // Attempt to parse new versioned format first
        if let Ok(mut session) = serde_json::from_str::<SessionFile>(&data) {
            match session.version {
                CURRENT_SESSION_VERSION => {
                    tracing::info!("Loaded session v{} from {}", session.version, path.display());
                    session.state.sandbox_root = std::env::current_dir()?;
                    return Ok(session.state);
                }
                v if v > CURRENT_SESSION_VERSION => {
                    return Err(anyhow::anyhow!("Session version {} is from a newer version of Gypsy. Please update.", v));
                }
                _ => {
                    tracing::info!("Migrating session from v{} to v{}", session.version, CURRENT_SESSION_VERSION);
                    session.state.sandbox_root = std::env::current_dir()?;
                    // Future migration logic would go here
                    return Ok(session.state);
                }
            }
        }

        // Fallback: Handle legacy format (pre-versioning)
        if let Ok(mut legacy_state) = serde_json::from_str::<DeepAgentState>(&data) {
            tracing::info!("Migrating legacy (unversioned) session to v{}", CURRENT_SESSION_VERSION);
            legacy_state.sandbox_root = std::env::current_dir()?;
            return Ok(legacy_state);
        }

        // Last resort: backup and start fresh
        tracing::warn!("Failed to parse session {}. Moving to backup and starting fresh.", path.display());
        let backup_path = path.with_extension("json.corrupt");
        let _ = std::fs::copy(&path, &backup_path);
        
        Ok(DeepAgentState {
            session_id: session_id.to_string(),
            context: Arc::new(Context { items: vec![] }),
            sandbox_root: std::env::current_dir()?,
        })
    }

    pub fn save_session(&self, state: &DeepAgentState) -> Result<()> {
        let path = self.get_session_path(&state.session_id);
        let session = SessionFile {
            version: CURRENT_SESSION_VERSION,
            state: state.clone(),
        };
        
        let data = serde_json::to_string_pretty(&session)?;
        let mut temp_path = path.clone();
        temp_path.set_extension("tmp");
        
        std::fs::write(&temp_path, data)?;
        std::fs::rename(temp_path, path)?;
        Ok(())
    }

    pub fn save_metrics(&self, session_id: &str, metrics: &SessionMetrics) -> Result<()> {
        let path = self.sessions_dir.join(format!("metrics_{}.json", session_id));
        let data = serde_json::to_string_pretty(metrics)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn list_sessions(&self) -> Result<Vec<String>> {
        if !self.sessions_dir.exists() {
            return Ok(vec![]);
        }
        
        let entries = std::fs::read_dir(&self.sessions_dir)?;
        let mut ids = Vec::new();
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name().into_string().unwrap_or_default();
            if name.starts_with("session_") && (name.ends_with(".json") || name.ends_with(".session")) {
                let id = name.trim_start_matches("session_")
                    .trim_end_matches(".json")
                    .trim_end_matches(".session");
                ids.push(id.to_string());
            }
        }
        Ok(ids)
    }
}
