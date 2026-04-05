use ratatui::{
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};
use ringbuffer::{AllocRingBuffer, RingBuffer};

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};

const MAX_VISIBLE_LOGS: usize = 300; // Hard limit for performance optimization

#[derive(Clone, Debug)]
pub enum LogEntry {
    Info(String),
    Debug(String),
    Warn(String),
    Error(String),
    Trace(String),
    User(String),
    Gypsy(String),
}

pub struct AppState {
    pub log: AllocRingBuffer<LogEntry>,
    pub tokens: usize,
    pub context_size: usize,
    pub current_step: String,
    pub input_buffer: String,
    pub status: String,
    pub is_thinking: bool,
    pub counter: usize,
    
    // New UI State
    pub available_commands: Vec<String>,
    pub autocomplete_suggestions: Vec<String>,
    pub show_debug: bool,
    pub log_scroll: u16,
    pub system_log_scroll: u16,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            status: "Idle".to_string(),
            current_step: "Waiting for input".to_string(),
            log: AllocRingBuffer::new(1000),
            tokens: 0,
            context_size: 0,
            input_buffer: String::new(),
            is_thinking: false,
            counter: 0,
            available_commands: Vec::new(),
            autocomplete_suggestions: Vec::new(),
            show_debug: false,
            log_scroll: 0,
            system_log_scroll: 0,
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

    // 1. Interaction (Chat) Logs
    let chat_log = state.log.iter()
        .filter(|e| {
            match e {
                LogEntry::User(_) | LogEntry::Gypsy(_) | LogEntry::Error(_) => true,
                _ => false
            }
        });

    // 2. Systems (Under-the-Hood) Logs
    let system_log = state.log.iter()
        .filter(|e| {
            match e {
                LogEntry::Debug(_) | LogEntry::Trace(_) => state.show_debug,
                LogEntry::Info(msg) | LogEntry::Warn(msg) => {
                    // Include any log that isn't a direct chat entry
                    !msg.contains("User:") && !msg.contains("Gypsy:") && !msg.contains("ERROR:")
                },
                _ => false
            }
        });

    // Apply visibility limits for performance
    let chat_log = chat_log.take(MAX_VISIBLE_LOGS);
    let system_log = system_log.take(MAX_VISIBLE_LOGS);

    let mut chat_text = Text::default();
    for entry in chat_log {
        let line = match entry {
            LogEntry::User(msg) => Line::from(vec![
                Span::styled("User: ", Style::default().fg(Color::Cyan)),
                Span::raw(msg),
            ]),
            LogEntry::Gypsy(msg) => Line::from(vec![
                Span::styled("Gypsy: ", Style::default().fg(Color::Green)),
                Span::raw(msg),
            ]),
            LogEntry::Error(msg) => Line::from(vec![
                Span::styled("ERROR: ", Style::default().fg(Color::Red)),
                Span::raw(msg),
            ]),
            _ => unreachable!("Chat log should only contain User/Gypsy/Error"),
        };
        chat_text.lines.push(line);
    }

    let chat_title = if state.log_scroll > 0 {
        format!("Agent Session Log [Scroll: {}]", state.log_scroll)
    } else {
        "Agent Session Log".to_string()
    };

    let chat_paragraph = Paragraph::new(chat_text)
        .block(Block::default().borders(Borders::ALL).title(chat_title))
        .wrap(Wrap { trim: true })
        .scroll((state.log_scroll, 0));
    
    f.render_widget(chat_paragraph, left_chunks[0]);

    let input_title = if state.is_thinking {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        format!("Thinking {} ", frames[state.counter % frames.len()])
    } else {
        "Input Prompt".to_string()
    };

    // Autocomplete layer
    if !state.autocomplete_suggestions.is_empty() {
        let suggestion_items: Vec<ListItem> = state.autocomplete_suggestions.iter()
            .map(|s| ListItem::new(s.as_str()))
            .collect();
        
        let suggestions_len = suggestion_items.len().min(5);
        let height = (suggestions_len as u16) + 2;
        
        let area = left_chunks[1];
        let popup_area = ratatui::layout::Rect::new(
            area.x,
            area.y.saturating_sub(height),
            area.width,
            height
        );
        
        let suggestions_list = List::new(suggestion_items)
            .block(Block::default().borders(Borders::ALL).title("Suggestions")
            .border_style(Style::default().fg(Color::Cyan)));
        
        f.render_widget(ratatui::widgets::Clear, popup_area);
        f.render_widget(suggestions_list, popup_area);
    }

    let input = Paragraph::new(state.input_buffer.as_str())
        .block(Block::default().borders(Borders::ALL).title(input_title));
    f.render_widget(input, left_chunks[1]);

    // Right Column: Dashboard
    let right_constraints = if state.show_debug {
        vec![
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(5), // Log panel takes middle space
            Constraint::Length(5),
        ]
    } else {
        vec![
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0), // Help Area will expand
            Constraint::Length(5),
        ]
    };

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(right_constraints)
        .split(chunks[1]);

    let status_style = if state.is_thinking {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
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

    if state.show_debug {
        // Render Under-the-Hood Logs
        let mut system_text = Text::default();
        for entry in system_log {
            let line = match entry {
                LogEntry::Warn(msg) => Line::from(vec![
                    Span::styled("WARN:  ", Style::default().fg(Color::Yellow)),
                    Span::raw(msg),
                ]),
                LogEntry::Debug(msg) => Line::from(vec![
                    Span::styled("DEBUG: ", Style::default().fg(Color::DarkGray)),
                    Span::raw(msg),
                ]),
                LogEntry::Trace(msg) => Line::from(vec![
                    Span::styled("TRACE: ", Style::default().fg(Color::Gray)),
                    Span::raw(msg),
                ]),
                LogEntry::Info(msg) => Line::from(vec![
                    Span::styled("INFO:  ", Style::default().fg(Color::White)),
                    Span::raw(msg),
                ]),
                _ => continue,
            };
            system_text.lines.push(line);
        }

        let system_title = if state.system_log_scroll > 0 {
            format!("Under-the-Hood [Scroll: {}] [DEBUG ON]", state.system_log_scroll)
        } else {
            "Under-the-Hood [DEBUG ON]".to_string()
        };

        let system_paragraph = Paragraph::new(system_text)
            .block(Block::default().borders(Borders::ALL).title(system_title))
            .wrap(Wrap { trim: true })
            .scroll((state.system_log_scroll, 0));
        
        f.render_widget(system_paragraph, right_chunks[4]);
    }

    // Help Area (Always the bottom-most chunk)
    let help_area = *right_chunks.last().unwrap();
    let help_columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(help_area);

    let (left_help, right_help) = if state.input_buffer.starts_with('/') {
        (
            Line::from(vec![
                Span::styled("Enter: ", Style::default().fg(Color::DarkGray)),
                Span::raw("Select"),
            ]),
            Line::from(vec![
                Span::styled("Tab: ", Style::default().fg(Color::DarkGray)),
                Span::raw("Complete"),
            ]),
        )
    } else {
        (
            Line::from(vec![
                Span::styled("Esc: ", Style::default().fg(Color::DarkGray)),
                Span::raw("Quit"),
            ]),
            Line::from(vec![
                Span::styled("PgUp: ", Style::default().fg(Color::DarkGray)),
                Span::raw("Up"),
            ]),
        )
    };

    let help_para_left = Paragraph::new(Text::from(vec![
        left_help,
        Line::from(vec![
            Span::styled("Enter: ", Style::default().fg(Color::DarkGray)),
            Span::raw("Submit"),
        ]),
        Line::from(vec![
            Span::styled("F1: ", Style::default().fg(Color::DarkGray)),
            Span::raw("Debug"),
        ]),
    ]))
    .block(Block::default().borders(Borders::ALL).title("Quick Help"));

    let help_para_right = Paragraph::new(Text::from(vec![
        right_help,
        Line::from(vec![
            Span::styled("PgDn: ", Style::default().fg(Color::DarkGray)),
            Span::raw("Down"),
        ]),
    ]))
    .block(Block::default().borders(Borders::ALL).title(""));

    f.render_widget(help_para_left, help_columns[0]);
    f.render_widget(help_para_right, help_columns[1]);
}
