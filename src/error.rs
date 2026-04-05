use std::io;

#[derive(Debug, thiserror::Error)]
pub enum GypsyError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Agent error: {0}")]
    AgentError(String),
    
    #[error("Tool execution error: {0}")]
    ToolError(String),
    
    #[error("Session error: {0}")]
    SessionError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
    
    #[error("General error: {0}")]
    General(#[from] anyhow::Error),
}

pub type GypsyResult<T> = Result<T, GypsyError>;
