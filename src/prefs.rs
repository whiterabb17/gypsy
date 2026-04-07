use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use anyhow::Result;

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct AppPrefs {
    pub disabled_mcps: HashSet<String>,
}

pub struct PrefsManager {
    path: PathBuf,
    prefs: AppPrefs,
}

impl PrefsManager {
    pub fn new(agent_dir: &Path) -> Self {
        let path = agent_dir.join("prefs.json");
        let prefs = if path.exists() {
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|content| serde_json::from_str(&content).ok())
                .unwrap_or_default()
        } else {
            AppPrefs::default()
        };

        Self { path, prefs }
    }

    pub fn is_mcp_enabled(&self, name: &str) -> bool {
        !self.prefs.disabled_mcps.contains(name)
    }

    pub fn set_mcp_enabled(&mut self, name: &str, enabled: bool) -> Result<()> {
        if enabled {
            self.prefs.disabled_mcps.remove(name);
        } else {
            self.prefs.disabled_mcps.insert(name.to_string());
        }
        self.save()
    }

    pub fn list_disabled_mcps(&self) -> Vec<String> {
        self.prefs.disabled_mcps.iter().cloned().collect()
    }

    fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(&self.prefs)?;
        std::fs::write(&self.path, content)?;
        Ok(())
    }
}
