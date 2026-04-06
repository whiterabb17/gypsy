use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;
use crate::error::GypsyResult;

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
            max_files: 5000,
            max_file_size: 10 * 1024 * 1024, // 10MB
            max_total_bytes: 500 * 1024 * 1024, // 500MB
            timeout_per_file: Duration::from_secs(5),
            max_depth: 20,
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

    pub fn walk_dir_safe(&self, start_dir: &Path) -> GypsyResult<Vec<PathBuf>> {
        let mut files = Vec::new();
        // Since std::fs::Metadata id() is unstable in standard library sometimes without specific os traits,
        // we map it via standard hashsets over specific IDs if posix, but on windows/cross-platform, 
        // canonicalize paths to detect symlink loops.
        let mut visited = HashSet::new();

        self._walk_dir_recursive(start_dir, &mut files, &mut visited, 0)?;
        Ok(files)
    }

    fn _walk_dir_recursive(
        &self,
        dir: &Path,
        files: &mut Vec<PathBuf>,
        visited: &mut HashSet<PathBuf>,
        depth: usize,
    ) -> GypsyResult<()> {
        if depth > self.config.max_depth {
            tracing::warn!("Max directory depth reached at {}", dir.display());
            return Ok(());
        }

        if files.len() >= self.config.max_files {
            tracing::warn!("Max file count reached in directory walk");
            return Ok(());
        }

        let canonical = std::fs::canonicalize(dir)
            .unwrap_or_else(|_| dir.to_path_buf());

        if visited.contains(&canonical) {
            tracing::debug!("Skipping visited directory loop: {}", dir.display());
            return Ok(());
        }
        visited.insert(canonical);

        for entry in std::fs::read_dir(dir)? {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::debug!("Failed to read directory entry in {}: {}", dir.display(), e);
                    continue;
                }
            };
            
            let file_type = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };

            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            
            if name == "target" || name == "node_modules" || name == ".git" || name.starts_with('.') {
                continue;
            }

            if file_type.is_file() {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
                if self.config.allowed_extensions.contains(ext) {
                    files.push(path);
                }
            } else if file_type.is_dir() {
                let _ = self._walk_dir_recursive(&path, files, visited, depth + 1);
            }
        }

        Ok(())
    }

    pub async fn consume_with_limits(&self, root: &Path) -> GypsyResult<ConsumeReport> {
        let mut report = ConsumeReport::default();
        let mut total_bytes = 0u64;
    
        let entries = self.walk_dir_safe(root)?;
        
        for path in entries.iter().take(self.config.max_files) {
            if total_bytes > self.config.max_total_bytes {
                report.truncated = true;
                break;
            }
            
            match tokio::time::timeout(
                self.config.timeout_per_file,
                tokio::fs::read_to_string(&path),
            ).await {
                Ok(Ok(content)) => {
                    let bytes = content.len() as u64;
                    if bytes > self.config.max_file_size {
                        report.skipped_oversized.push(path.clone());
                        continue;
                    }
                    
                    total_bytes += bytes;
                    report.processed_files.push(path.clone());
                    report.total_bytes += bytes;
                }
                Ok(Err(e)) => {
                    tracing::warn!("Failed to read {}: {}", path.display(), e);
                    report.read_errors.push((path.clone(), e.to_string()));
                }
                Err(_) => {
                    tracing::warn!("Timeout reading {}", path.display());
                    report.timeouts.push(path.clone());
                }
            }
        }
        
        Ok(report)
    }
}
