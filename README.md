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
- **📦 Staging Vault Architecture**: All file modifications are staged in a secure vault; changes only touch your project root after manual approval.
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

---

## ⌨️ Operational Commands

Gypsy isn't just a chatbot—it's an operational workbench. Use these internal commands to manage your environment:

| Command | Action |
| :--- | :--- |
| `/session list` | Lists all saved sessions (including dynamically timestamped ones). |
| `/session switch <id>` | Instantly reloads the agent with the history and context of another session. |
| `/consume` | **Study Mode**: Recursively walk the project directory and index it into local knowledge. |
| `/review` | **Audit Mode**: Performs an AI review of all files currently staged in the **Vault**. |

---

## 🏛️ Architecture

Gypsy follows the **DeepAgent** methodology: **Agent = Model + Harness + Memory**.

- **Model**: Unified provider support (Ollama, Anthropic, etc.).
- **Harness**: [**Mentalist**](https://github.com/whiterabb17/mentalist) provides the secure execution loop and safety gates.
- **Memory**: [**MindPalace**](https://github.com/whiterabb17/mindpalace) manages the 7-layer context optimization and long-term knowledge base.

---

## 🤝 Repositories in the Ecosystem

- **[Mentalist](https://github.com/whiterabb17/mentalist)**: High-performance agent harness and executor tools.
- **[MindPalace](https://github.com/whiterabb17/mindpalace)**: State-of-the-art context optimization and memory retrieval inspired by the claude memory layers.

---

## 📝 License

Gypsy is released under the MIT License. Built with ❤️ by [whiterabb17](https://github.com/whiterabb17).
