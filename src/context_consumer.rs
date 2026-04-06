use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;
use crate::error::GypsyResult;
use ignore::WalkBuilder;

pub struct ContextConsumerConfig {
    pub max_files: usize,
    pub max_file_size: u64,
    pub max_total_bytes: u64,
    pub timeout_per_file: Duration,
    pub max_depth: usize,
    pub allowed_extensions: HashSet<String>,
}

impl Default for ContextConsumerConfig {
    fn default() -> Self {
        let mut allowed = HashSet::new();
        allowed.extend(["rs", "toml", "md", "txt", "js", "ts", "json", "env", "py", "go", "cpp", "c", "h", "hpp"].iter().map(|s| s.to_string()));
        Self {
            max_files: 1000,
            max_file_size: 2 * 1024 * 1024, // 2MB (Reduced for safety)
            max_total_bytes: 100 * 1024 * 1024, // 100MB
            timeout_per_file: Duration::from_secs(2),
            max_depth: 10, // Reduced for safety
            allowed_extensions: allowed,
        }
    }
}

#[derive(Debug, Default)]
pub struct ConsumeReport {
    pub processed_files: Vec<PathBuf>,
    pub skipped_oversized: Vec<PathBuf>,
    pub read_errors: Vec<(PathBuf, String)>,
    pub timeouts: Vec<PathBuf>,
    pub total_bytes: u64,
    pub truncated: bool,
}

pub struct ContextConsumer {
    pub config: ContextConsumerConfig,
}

impl ContextConsumer {
    pub fn new() -> Self {
        Self {
            config: ContextConsumerConfig::default(),
        }
    }

    pub fn walk_dir_hardened(&self, root: &Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        // Use ignore crate for robust .gitignore and .gypsyignore handling
        let walker = WalkBuilder::new(root)
            .hidden(true)
            .git_ignore(true)
            .add_custom_ignore_filename(".gypsyignore")
            .max_depth(Some(self.config.max_depth))
            .build();

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();
            if !path.is_file() { continue; }

            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
            if self.config.allowed_extensions.contains(ext) {
                files.push(path.to_path_buf());
            }

            if files.len() >= self.config.max_files { break; }
        }
        files
    }

    pub async fn consume_with_limits(&self, root: &Path) -> GypsyResult<ConsumeReport> {
        let mut report = ConsumeReport::default();
        let mut total_bytes = 0u64;
    
        let entries = self.walk_dir_hardened(root);
        
        for path in entries {
            if total_bytes > self.config.max_total_bytes {
                report.truncated = true;
                break;
            }
            
            let metadata = match tokio::fs::metadata(&path).await {
                Ok(m) => m,
                Err(_) => continue,
            };

            if metadata.len() > self.config.max_file_size {
                report.skipped_oversized.push(path);
                continue;
            }

            match tokio::time::timeout(
                self.config.timeout_per_file,
                tokio::fs::read_to_string(&path),
            ).await {
                Ok(Ok(content)) => {
                    let bytes = content.len() as u64;
                    total_bytes += bytes;
                    report.processed_files.push(path);
                    report.total_bytes += bytes;
                }
                Ok(Err(e)) => {
                    report.read_errors.push((path, e.to_string()));
                }
                Err(_) => {
                    report.timeouts.push(path);
                }
            }
        }
        
        Ok(report)
    }
}
