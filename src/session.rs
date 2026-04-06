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

pub const CURRENT_SESSION_VERSION: u32 = 1;

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
            return Ok(DeepAgentState {
                session_id: session_id.to_string(),
                context: Arc::new(Context { items: vec![] }),
                sandbox_root: std::env::current_dir()?,
            });
        }

        let data = std::fs::read_to_string(&path)?;
        match serde_json::from_str::<SessionFile>(&data) {
            Ok(mut session) => {
                tracing::info!("Loaded session version {} from {}", session.version, path.display());
                session.state.sandbox_root = std::env::current_dir()?;
                Ok(session.state)
            }
            Err(parse_err) => {
                // Migration logic: handle legacy format
                if let Ok(mut legacy_state) = serde_json::from_str::<DeepAgentState>(&data) {
                    tracing::info!("Migrating legacy session to version {}", CURRENT_SESSION_VERSION);
                    legacy_state.sandbox_root = std::env::current_dir()?;
                    Ok(legacy_state)
                } else {
                    tracing::warn!("Failed to parse session {}: {}. Creating backup.", path.display(), parse_err);
                    let backup_path = path.with_extension("json.corrupt");
                    let _ = std::fs::copy(&path, &backup_path);
                    Ok(DeepAgentState {
                        session_id: session_id.to_string(),
                        context: Arc::new(Context { items: vec![] }),
                        sandbox_root: std::env::current_dir()?,
                    })
                }
            }
        }
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
