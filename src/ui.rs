use ratatui::{
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};
// use std::time::Duration;

#[derive(Default)]
pub struct AppState {
    pub log: Vec<String>,
    pub tokens: usize,
    pub context_size: usize,
    pub current_step: String,
    pub input_buffer: String,
    pub status: String,
    pub is_thinking: bool,
    pub counter: usize,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            status: "Idle".to_string(),
            current_step: "Waiting for input".to_string(),
            log: Vec::new(),
            tokens: 0,
            context_size: 0,
            input_buffer: String::new(),
            is_thinking: false,
            counter: 0,
        }
    }
}

pub fn ui(f: &mut Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(f.size());

    // Left Column: Interaction Log & Input
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(chunks[0]);

    let log_items: Vec<ListItem> = state
        .log
        .iter()
        .map(|line| ListItem::new(line.as_str()))
        .collect();
    let log_list = List::new(log_items)
        .block(Block::default().borders(Borders::ALL).title("Agent Session Log"));
    f.render_widget(log_list, left_chunks[0]);

    let input_title = if state.is_thinking {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        format!("Thinking {} ", frames[state.counter % frames.len()])
    } else {
        "Input Prompt".to_string()
    };

    let input = Paragraph::new(state.input_buffer.as_str())
        .block(Block::default().borders(Borders::ALL).title(input_title));
    f.render_widget(input, left_chunks[1]);

    // Right Column: Dashboard
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(chunks[1]);

    let status_style = if state.is_thinking {
        ratatui::style::Style::default().fg(ratatui::style::Color::Yellow)
    } else {
        ratatui::style::Style::default()
    };

    let status_para = Paragraph::new(state.status.as_str())
        .style(status_style)
        .block(Block::default().borders(Borders::ALL).title("System Status"));
    f.render_widget(status_para, right_chunks[0]);

    let step_para = Paragraph::new(state.current_step.as_str())
        .block(Block::default().borders(Borders::ALL).title("Current Phase"));
    f.render_widget(step_para, right_chunks[1]);

    let token_para = Paragraph::new(format!("{}", state.tokens))
        .block(Block::default().borders(Borders::ALL).title("Total Tokens Used"));
    f.render_widget(token_para, right_chunks[2]);

    let context_para = Paragraph::new(format!("{}", state.context_size))
        .block(Block::default().borders(Borders::ALL).title("Context Items Count"));
    f.render_widget(context_para, right_chunks[3]);

    let help = Paragraph::new("Esc: Quit\nEnter: Submit Prompt")
        .block(Block::default().borders(Borders::ALL).title("Quick Help"));
    f.render_widget(help, right_chunks[4]);
}
