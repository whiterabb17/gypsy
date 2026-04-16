use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap, Gauge, Sparkline},
    Frame,
};
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::collections::HashMap;

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
    System(String),
    Reasoning {
        main_step: String,
        sub_steps: Vec<String>,
        collapsed: bool,
    },
}

pub struct AppState {
    pub log: AllocRingBuffer<LogEntry>,
    pub tokens: usize,
    pub max_tokens: usize,
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
    pub follow_chat: bool,
    pub follow_system: bool,
    
    // Detailed Metrics
    pub tool_calls_total: usize,
    pub last_tool_name: String,
    pub last_tool_duration_ms: u128,
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
    pub llm_latency_ms: u128,
    
    // New UX State
    pub progress: Option<f32>,
    pub phase_progress: Option<f32>,
    pub context_status: HashMap<String, bool>, // Renamed from tool_status for clarity? No, keep tool_status
    pub tool_status: HashMap<String, bool>,
    pub context_history: Vec<u64>,
    pub rolling_latency: Vec<u128>,
    pub fallback_pending: bool,
    
    // Internal Tracking
    pub last_input_tokens: usize,
    pub last_output_tokens: usize,
    pub pending_plan: Option<mentalist::mem_planner::ExecutionPlan>,
    pub active_plan: Option<mentalist::mem_planner::ExecutionPlan>,
    pub completed_tasks: std::collections::HashSet<mentalist::mem_planner::TaskId>,
    pub reasoning_rects: HashMap<usize, Rect>, // Map log index to clickable Rect
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    pub fn new() -> Self {
        Self {
            status: "Idle".to_string(),
            current_step: "Waiting for input".to_string(),
            log: AllocRingBuffer::new(1000),
            tokens: 0,
            max_tokens: 0,
            context_size: 0,
            input_buffer: String::new(),
            is_thinking: false,
            counter: 0,
            available_commands: Vec::new(),
            autocomplete_suggestions: Vec::new(),
            show_debug: false,
            log_scroll: 0,
            system_log_scroll: 0,
            follow_chat: true,
            follow_system: true,
            tool_calls_total: 0,
            last_tool_name: "None".to_string(),
            last_tool_duration_ms: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            llm_latency_ms: 0,
            progress: None,
            phase_progress: None,
            context_status: HashMap::new(),
            tool_status: HashMap::new(),
            context_history: Vec::new(),
            rolling_latency: Vec::with_capacity(5),
            fallback_pending: false,
            last_input_tokens: 0,
            last_output_tokens: 0,
            pending_plan: None,
            active_plan: None,
            completed_tasks: std::collections::HashSet::new(),
            reasoning_rects: HashMap::new(),
        }
    }
}

fn estimate_height(text: &Text, width: u16) -> u16 {
    if width == 0 { return 0; }
    let mut height = 0;
    for line in &text.lines {
        let line_width = line.width() as u16;
        if line_width == 0 {
            height += 1;
        } else {
            // Basic wrap estimation: total width / available width
            height += line_width.div_ceil(width);
        }
    }
    height
}

pub fn ui(f: &mut Frame, state: &mut AppState) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(f.size());

    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(1), // Context Bar (Always)
            Constraint::Length(if state.progress.is_some() { 1 } else { 0 }), // Progress Bar (Optional)
        ])
        .split(chunks[0]);

    // Calculate input height based on content
    let input_text = Text::raw(state.input_buffer.as_str());
    let input_width = main_chunks[0].width.saturating_sub(2);
    let input_height = estimate_height(&input_text, input_width).clamp(1, 15) + 2;

    // 1. Interaction (Chat) Logs
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(input_height)])
        .split(main_chunks[0]);

    // 1. Interaction (Chat) Logs
    let chat_log = state.log.iter().enumerate()
        .filter(|(_, e)| {
            matches!(e, LogEntry::User(_) | LogEntry::Gypsy(_) | LogEntry::Error(_) | LogEntry::System(_) | LogEntry::Reasoning {..})
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

    state.reasoning_rects.clear();
    let chat_inner_width = left_chunks[0].width.saturating_sub(2);
    let chat_inner_height = left_chunks[0].height.saturating_sub(2);
    let mut current_line_offset: u16 = 0;
    let mut chat_text = Text::default();

    for (idx, entry) in chat_log {
        let mut entry_lines = Vec::new();
        match entry {
            LogEntry::User(msg) => {
                entry_lines.push(Line::from(vec![
                    Span::styled("User: ", Style::default().fg(Color::Cyan)),
                    Span::raw(msg),
                ]));
            }
            LogEntry::Gypsy(msg) => {
                entry_lines.push(Line::from(vec![
                    Span::styled("Gypsy: ", Style::default().fg(Color::Green)),
                    Span::raw(msg),
                ]));
            }
            LogEntry::Error(msg) => {
                entry_lines.push(Line::from(vec![
                    Span::styled("ERROR: ", Style::default().fg(Color::Red)),
                    Span::raw(msg),
                ]));
            }
            LogEntry::System(msg) => {
                entry_lines.push(Line::from(vec![
                    Span::styled("SYSTEM: ", Style::default().fg(Color::DarkGray)),
                    Span::raw(msg),
                ]));
            }
            LogEntry::Reasoning { main_step, sub_steps, collapsed } => {
                let arrow = if *collapsed { "▶ " } else { "▼ " };
                let header = Line::from(vec![
                    Span::styled(arrow, Style::default().fg(Color::Yellow)),
                    Span::styled(main_step, Style::default().fg(Color::Yellow).add_modifier(ratatui::style::Modifier::ITALIC)),
                ]);
                
                // Track Rect for this reasoning entry
                // Only if it's within the visible log scroll? No, track all, we'll hit test with scroll.
                let _entry_height = if *collapsed {
                    estimate_height(&Text::from(header.clone()), chat_inner_width)
                } else {
                    let mut text = Text::from(header.clone());
                    for step in sub_steps {
                        text.lines.push(Line::from(vec![
                            Span::raw("  "),
                            Span::styled(step, Style::default().fg(Color::DarkGray).add_modifier(ratatui::style::Modifier::ITALIC)),
                        ]));
                    }
                    estimate_height(&text, chat_inner_width)
                };

                let rect = Rect::new(
                    left_chunks[0].x + 1,
                    left_chunks[0].y + 1 + current_line_offset.saturating_sub(state.log_scroll),
                    left_chunks[0].width.saturating_sub(2),
                    1 // Just the header is clickable
                );
                
                // We only store it if it's actually visible on screen
                if current_line_offset >= state.log_scroll && (current_line_offset - state.log_scroll) < chat_inner_height {
                    state.reasoning_rects.insert(idx, rect);
                }

                entry_lines.push(header);
                if !*collapsed {
                    for step in sub_steps {
                        entry_lines.push(Line::from(vec![
                            Span::raw("  "),
                            Span::styled(step, Style::default().fg(Color::DarkGray).add_modifier(ratatui::style::Modifier::ITALIC)),
                        ]));
                    }
                }
            }
            _ => continue,
        };

        let entry_text = Text::from(entry_lines.clone());
        let entry_height = estimate_height(&entry_text, chat_inner_width);
        
        chat_text.lines.extend(entry_lines);
        current_line_offset += entry_height;
    }

    let chat_content_height = estimate_height(&chat_text, chat_inner_width);

    if state.follow_chat {
        state.log_scroll = chat_content_height
            .saturating_sub(chat_inner_height)
            .min(chat_content_height);
    }

    let chat_title = if state.log_scroll > 0 {
        format!(" Agent Session Log [Scroll: {}] ", state.log_scroll)
    } else {
        " Agent Session Log ".to_string()
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
        .block(Block::default()
            .borders(Borders::ALL)
            .title(input_title)
            .border_style(if state.fallback_pending { Style::default().fg(Color::LightMagenta) } else { Style::default() }))
        .wrap(Wrap { trim: true });
    f.render_widget(input, left_chunks[1]);

    // Global Context Usage Bar (Utilization of the 32k window)
    let context_ratio = if state.max_tokens > 0 {
        (state.context_size as f64 / state.max_tokens as f64).min(1.0)
    } else {
        0.0
    };

    let context_color = if context_ratio >= 0.8 {
        Color::Red
    } else if context_ratio >= 0.6 {
        Color::Yellow
    } else {
        Color::Cyan
    };

    let context_label = format!(
        " Context: {}/{} ({:.1}%) ",
        state.context_size,
        state.max_tokens,
        context_ratio * 100.0
    );

    let context_gauge = Gauge::default()
        .block(Block::default())
        .gauge_style(Style::default().fg(context_color).bg(Color::Black))
        .ratio(context_ratio)
        .label(context_label);
    
    f.render_widget(context_gauge, main_chunks[1]);

    // Global Task Progress Bar (Rendered below context if active)
    if let Some(progress) = state.progress {
        let gauge = Gauge::default()
            .block(Block::default())
            .gauge_style(Style::default().fg(Color::Magenta).bg(Color::Black))
            .ratio(progress as f64);
        f.render_widget(gauge, main_chunks[2]);
    }

    // Right Column: Dashboard
    let right_constraints = if state.show_debug {
        vec![
            Constraint::Length(3), // Diagnostics
            Constraint::Length(3), // Token Usage
            Constraint::Length(3), // Capacity & Tools
            Constraint::Length(8), // Tasks Box
            Constraint::Min(5),    // Under-the-Hood
            Constraint::Length(5), // Quick Help
        ]
    } else {
        vec![
            Constraint::Length(3), // Diagnostics
            Constraint::Length(3), // Token Usage
            Constraint::Length(3), // Capacity & Tools
            Constraint::Min(0),    // Tasks Box (expands)
            Constraint::Length(0), // Under-the-Hood (hidden)
            Constraint::Length(5), // Quick Help
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

    // 1. Diagnostics Panel (System + LLM)
    let diag_text = vec![
        Line::from(vec![
            Span::styled("Status: ", Style::default().fg(Color::Gray)),
            Span::styled(state.status.as_str(), status_style),
        ]),
        Line::from(vec![
            Span::styled("Phase:  ", Style::default().fg(Color::Gray)),
            Span::raw(state.current_step.as_str()),
        ]),
        Line::from(vec![
            Span::styled("Latency: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}ms", state.llm_latency_ms), Style::default().fg(Color::Yellow)),
        ]),
    ];
    let diag_para = Paragraph::new(diag_text)
        .block(Block::default().borders(Borders::ALL).title(" Diagnostics "));
    f.render_widget(diag_para, right_chunks[0]);

    // Phase Progress in Diagnostics
    if let Some(p) = state.phase_progress {
        let gauge_area = Rect::new(right_chunks[0].x + 1, right_chunks[0].y + 2, right_chunks[0].width - 2, 1);
        let phase_gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Yellow))
            .ratio(p as f64)
            .label("");
        f.render_widget(phase_gauge, gauge_area);
    }

    // 2. Token Metrics
    let token_text = vec![
        Line::from(vec![
            Span::styled("Total Service: ", Style::default().fg(Color::Gray)),
            Span::raw(format!("{}", state.total_input_tokens + state.total_output_tokens)),
        ]),
        Line::from(vec![
            Span::styled("In:    ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", state.total_input_tokens), Style::default().fg(Color::Blue)),
            Span::styled(" | Out: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", state.total_output_tokens), Style::default().fg(Color::Green)),
        ]),
    ];
    let token_para = Paragraph::new(token_text)
        .block(Block::default().borders(Borders::ALL).title(" Token Usage "));
    f.render_widget(token_para, right_chunks[1]);

    // 3. Capacity & Tools (Always shown)
    let mut mem_text = vec![
        Line::from(vec![
            Span::styled("Context Size: ", Style::default().fg(Color::Gray)),
            Span::raw(format!("{}", state.context_size)),
        ]),
        Line::from(vec![
            Span::styled("Tool Count:   ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", state.tool_calls_total), Style::default().fg(Color::Magenta)),
        ]),
    ];

    // Show last 2 tool results
    let mut recent_tools: Vec<_> = state.tool_status.iter().collect();
    recent_tools.sort_by(|a, b| a.0.cmp(b.0)); 
    for (name, success) in recent_tools.iter().take(2) {
        let color = if **success { Color::Green } else { Color::Red };
        let icon = if **success { "✓" } else { "✗" };
        mem_text.push(Line::from(vec![
            Span::styled(format!(" {} ", icon), Style::default().fg(color)),
            Span::styled((*name).clone(), Style::default().fg(Color::Gray)),
        ]));
    }

    let mem_para = Paragraph::new(mem_text)
        .block(Block::default().borders(Borders::ALL).title(" Capacity & Tools "));
    f.render_widget(mem_para, right_chunks[2]);
    
    // Sparkline for context history
    if !state.context_history.is_empty() {
        let sparkline_area = Rect::new(right_chunks[2].x + 1, right_chunks[2].y + 1, right_chunks[2].width - 2, 1);
        let sparkline = Sparkline::default()
            .data(&state.context_history)
            .style(Style::default().fg(Color::Blue));
        f.render_widget(sparkline, sparkline_area);
    }

    // 4. Tasks Box (Active or Pending Plan)
    let mut task_text = Vec::new();
    let mut task_title = " Tasks Box ".to_string();

    if let Some(ref plan) = state.pending_plan {
        task_title = " Plan Review (PENDING) ".to_string();
        task_text.push(Line::from(vec![
            Span::styled("PLAN REQUIRES APPROVAL", Style::default().fg(Color::Cyan).add_modifier(ratatui::style::Modifier::BOLD)),
        ]));
        task_text.push(Line::from(vec![
            Span::styled(format!("Content: {:.50}...", plan.content), Style::default().fg(Color::Gray)),
        ]));
        task_text.push(Line::from(""));

        let mut tasks: Vec<_> = plan.tasks.values().collect();
        tasks.sort_by(|a, b| a.id.0.cmp(&b.id.0)); // Simple sort by ID
        for task in tasks {
            let status_icon = " [ ] ";
            task_text.push(Line::from(vec![
                Span::styled(status_icon, Style::default().fg(Color::Yellow)),
                Span::raw(&task.name),
            ]));
        }
    } else if let Some(ref plan) = state.active_plan {
        task_title = " Active Plan ".to_string();
        let mut tasks: Vec<_> = plan.tasks.values().collect();
        tasks.sort_by(|a, b| a.id.0.cmp(&b.id.0));
        
        for task in tasks {
            let (status_icon, color) = if state.completed_tasks.contains(&task.id) {
                (" [✓] ", Color::Green)
            } else {
                (" [ ] ", Color::Yellow)
            };
            
            task_text.push(Line::from(vec![
                Span::styled(status_icon, Style::default().fg(color)),
                Span::raw(&task.name),
            ]));
        }
    } else {
        task_text.push(Line::from(vec![
            Span::styled("No active plan.", Style::default().fg(Color::DarkGray)),
        ]));
    }

    let task_para = Paragraph::new(task_text)
        .block(Block::default().borders(Borders::ALL).title(task_title))
        .wrap(Wrap { trim: true });
    f.render_widget(task_para, right_chunks[3]);

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

        let system_inner_width = right_chunks[4].width.saturating_sub(2);
        let system_inner_height = right_chunks[4].height.saturating_sub(2);
        let system_content_height = estimate_height(&system_text, system_inner_width);

        if state.follow_system {
            state.system_log_scroll = system_content_height
                .saturating_sub(system_inner_height)
                .min(system_content_height);
        }

        let system_title = if state.system_log_scroll > 0 {
            format!(" Under-the-Hood [Scroll: {}] [DEBUG ON] ", state.system_log_scroll)
        } else {
            " Under-the-Hood [DEBUG ON] ".to_string()
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
        Line::from(vec![
            Span::styled("Ctrl-Y: ", Style::default().fg(Color::DarkGray)),
            Span::raw("Copy"),
        ]),
    ]))
    .block(Block::default().borders(Borders::ALL).title("Quick Help"));

    let help_para_right = Paragraph::new(Text::from(vec![
        right_help,
        Line::from(vec![
            Span::styled("PgDn: ", Style::default().fg(Color::DarkGray)),
            Span::raw("Down"),
        ]),
        Line::from(vec![
            Span::styled("Shift+Sel: ", Style::default().fg(Color::Yellow)),
            Span::raw("Native Copy"),
        ]),
    ]))
    .block(Block::default().borders(Borders::ALL).title(""));

    f.render_widget(help_para_left, help_columns[0]);
    f.render_widget(help_para_right, help_columns[1]);

    // Fallback Confirmation Overlay
    if state.fallback_pending {
        let area = Rect::new(f.size().width / 4, f.size().height / 2 - 2, f.size().width / 2, 5);
        let block = Block::default().borders(Borders::ALL).title(" Fallback Confirmation ").border_style(Style::default().fg(Color::LightMagenta));
        let text = Text::from(vec![
            Line::from(vec![Span::raw("Primary provider failed. Fallback to Ollama?")]),
            Line::from(vec![Span::styled("  [Y] Yes / [N] No", Style::default().fg(Color::Yellow))]),
        ]);
        let para = Paragraph::new(text).block(block).wrap(Wrap { trim: true });
        f.render_widget(ratatui::widgets::Clear, area);
        f.render_widget(para, area);
    }
}
