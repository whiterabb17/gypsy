# 🎩 Gypsy: The Hardened, Stateful TUI Agent

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-macos%20%7C%20linux%20%7C%20windows-lightgrey.svg)]()
[![State: Production-Ready](https://img.shields.io/badge/state-production--ready-blue.svg)]()

**Gypsy** is a state-of-the-art Terminal User Interface (TUI) agent designed for high-stakes, security-first automation. Built on the [**Mentalist**](https://github.com/whiterabb17/mentalist) harness and powered by the [**MindPalace**](https://github.com/whiterabb17/mindpalace) memory ecosystem, Gypsy combines professional-grade security isolation with real-time, stateful reasoning.

---

## ⚡ Key Features

- **🛡️ Hardened Execution**: Multi-layered sandbox support (Wasm/Docker) with strict **4GB RAM limits**, **CPU Fueling**, and a **Malicious Command Blocker**.
- **🧠 Resilient Long-Term Memory**: Automatic fact extraction and RAG-based context injection via the **MindPalace** core.
- **🌐 Built-in Web Search**: Integrated **Firecrawl** MCP support for advanced web scraping and markdown data retrieval.
- **📂 Secure Filesystem MCP**: Native, standard-compliant filesystem tools restricted to your project root or workspace.
- **📦 Staging Vault Architecture**: All file modifications are staged in a secure vault; changes only touch your project root after manual approval.
- **🧩 Extensible Skills**: Drop new Python/JS scripts into the `skills/` folder to instantly "teach" the agent new capabilities.
- **✨ Professional UI/UX**: Real-time LLM streaming, dynamic thinking visuals, and a session-centric TUI with comprehensive metrics.
- **🤖 Autonomous Infrastructure**: Cross-platform automation of **Ollama** life-cycles (auto-start, auto-pull, and health monitoring).

---

## 🚀 Getting Started

### 1. Prerequisites
- **Rust**: `rustc` 1.75+
- **Ollama** (Optional but recommended): Ensure [Ollama](https://ollama.com/) is installed for local-first execution.

### 2. Installation
Clone the repository and build from source:
```bash
git clone https://github.com/whiterabb17/gypsy
cd gypsy/gypsy
cargo build --release
```

### 3. Configuration
Copy the example environment file and tune it to your needs:
```bash
cp .env.example .env
```
Key settings in `.env`:
- `PROVIDER`: `ollama`, `openai`, `anthropic`, or `gemini`.
- `SANDBOX_MODE`: `wasm` (Safe) or `local` (Direct).
- `VAULT_PATH`: Staging directory for agent-modified files.
- `FIRECRAWL_API_KEY`: Required for web-search capabilities.
- `MCP_FS_PATHS`: Comma-separated list of local paths the agent can access.

---

## 🕹️ Interacting with Gypsy

Gypsy provides a powerful, keyboard-driven interface with several helpful shortcuts for efficient navigation:

| Key | Action |
| :--- | :--- |
| **`/`** | **Command Autocomplete**: Typing `/` opens a suggestions box for available commands and tools. |
| **`Tab`** | **Autocomplete**: Rapidly complete the first suggested command. |
| **`F1`** | **Toggle Debug**: Show/Hide detailed `TRACE` and `DEBUG` logs (useful for troubleshooting). |
| **`PgUp / PgDn`** | **Scroll**: Manually scroll through the session log history. |
| **`Esc`**| **Quit**: Cleanly exit the session. |
| **`Enter`** | **Submit**: Send your prompt to the agent. |

---

## ⌨️ Operational Commands

Gypsy isn't just a chatbot—it's an operational workbench. Use these internal commands to manage your environment:

| `/mcp list` | **Status Dashboard**: Real-time monitoring of all connected MCP servers and tools. |
| `/mcp enable <name>` | **Hot-Plug**: Instantly enables a previously disabled MCP server or toolset. |
| `/mcp disable <name>` | **Isolation**: Disables an MCP server, removing its tools from the agent's context. |
| `/session list` | Lists all saved sessions (including dynamically timestamped ones). |
| `/session switch <id>` | Instantly reloads the agent with the history and context of another session. |
| `/consume` | **Study Mode**: Recursively walk the project directory and index it into local knowledge. |
| `/review` | **Audit Mode**: Performs an AI review of all files currently staged in the **Vault**. |
| `/summarize` | **Context Reduction**: Triggers the 7-layer optimization loop to compact history. |
| `/exit` | **Graceful Shutdown**: Shuts down the agent and restores terminal state. |

> [!TIP]
> You can also press `Ctrl+C` or `Esc` at any time to gracefully exit and restore your terminal.

---

## 🏗️ Architecture: The DeepAgent Engine

Gypsy follows a strictly tiered **DeepAgent** methodology, where the agent is defined as:  
`Agent = Unified Model + Secure Harness + Multi-Layered Memory`.

### 1. Unified Provider Engine (Shared)
To ensure maximum resource efficiency, Gypsy deduplicates model provider instances. Whether the agent is reasoning, extracting facts, or calculating embeddings, it uses a shared, thread-safe `Arc`-wrapped provider engine.
- **Supported Providers**: Ollama (Local), Anthropic (Claude), OpenAI (GPT-4), Gemini (Pro/Ultra).
- **Auto-Lifecycle**: For Ollama, Gypsy automatically manages daemon startup, model pulling, and health-check monitoring with a 5s fail-safe timeout.

### 2. The 7-Layer Memory System (MindPalace)
Gypsy doesn't just "remember" text; it optimizes context using a 7-layer hierarchy inspired by high-stakes cognitive architectures:
- **L1-L3 (Cache/Working)**: Immediate conversation window.
- **L4-L6 (Summarized/Reflective)**: Heuristic-based compaction of older turns to preserve "intent" without token bloat.
- **L7 (Deep Knowledge)**: RAG-enabled retrieval from the `FileStorage` backend.

### 3. Session Lifecycle & Persistence
Gypsy treats every interaction as a durable mission. 
- **Versioning**: All sessions use standard `v1` JSON versioning to support seamless future migrations.
- **Atomic Persistence**: State is saved using a "Write-Temp-then-Rename" pattern to ensure a crash never corrupts your history.
- **Migration**: Gypsy automatically detects legacy `.session` files and migrates them to the new `.json` versioned standard on first load.

---

## 🛡️ Operational Hardening & Safety

Gypsy is designed to be "safe by default" for developers working in sensitive environments.

- **Staging Vault**: Any tool that attempts to modify your filesystem works in a **Vault** sub-directory first. You review the changes before they are synced to your root.
- **Resource Guardrails**: Built-in limits for tool execution (e.g., 4GB RAM ceiling, 32-argument command ceiling) prevent runaway processes.
- **Sandboxed Execution**: Choose between `local` (trusted) or `wasm` (isolated) execution modes for untrusted scripts.

---

## ⚙️ Advanced Configuration

Configure Gypsy via `.env` or system environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PROVIDER` | LLM backend (`ollama`, `openai`, etc.) | `ollama` |
| `MODEL_NAME` | The specific model ID to use | `llama3` |
| `SESSIONS_PATH` | Directory for saved agent states | `.agent/sessions` |
| `STORAGE_PATH` | Root for long-term vector/fact storage | `.agent/storage` |
| `VAULT_PATH` | Staging area for file modifications | `.agent/vault` |
| `OLLAMA_BASE_URL` | Endpoint for Ollama API | `http://localhost:11434` |
| `LOG_LEVEL` | Logging verbosity (`info`, `debug`, `trace`) | `info` |

---

## 🤝 Repositories in the Ecosystem

- **[Mentalist](https://github.com/whiterabb17/mentalist)**: The high-performance agent harness and secure executor engine.
- **[MindPalace](https://github.com/whiterabb17/mindpalace)**: The SOTA context optimization and memory retrieval library.

---

## 📝 License

Gypsy is released under the MIT License. Built with ❤️ for the open-source community by [whiterabb17](https://github.com/whiterabb17).
