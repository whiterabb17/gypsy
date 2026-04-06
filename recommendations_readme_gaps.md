A thorough audit of the `gypsy` README versus the actual codebase has identified discrepancies where features are claimed in the documentation but are missing, incomplete, or only partially implemented in the current repository. This issue tracks those gaps, proposes remediation plans, and surfaces any additional findings encountered during review.

---

## 1. Tavily MCP Web Search Support ✅ (Removed from Roadmap)
- **README Claim**: "Integrated Firecrawl and Tavily MCP support for advanced web scraping and markdown data retrieval."
- **Finding**: Only Firecrawl integration is present.
- **Action**: Per user feedback, Tavily support has been removed from the README instead of being implemented.

---

## 2. Professional UI/UX (Realtime LLM Streaming, Dynamic Thinking Visuals, Session-Centric TUI with Metrics) ⛔
- **README Claim**: "Real-time LLM streaming, dynamic thinking visuals, and a session-centric TUI with comprehensive metrics."
- **Findings**:
  - While a TUI is implemented (`src/ui.rs`, `ratatui`, `crossterm`), many promised visuals are minimal or absent.
  - "Dynamic thinking visuals": Only a simple spinner/indicator—not the rich visual or interactive feedback described in the README.
  - Metrics exist but are limited; e.g., system metrics, context visualization, or advanced data panels are not implemented.
- **Remediation Plan**:
  - Expand TUI to show more LLM internals, context windows, system health, and live streaming tokens if possible.
  - Implement richer "thinking" feedback (e.g., status animations, token-by-token update, visual reasoning traces).
  - Update README to indicate which UI/UX features are currently available and which remain future work.

---

## 3. Ollama Lifecycle Automation ⛔
- **README Claim**: "Cross-platform automation of Ollama life-cycles (auto-start, auto-pull, health monitoring)."
- **Findings**:
  - Some efforts at Ollama readiness are present (see `service_manager.ensure_ollama_ready`), however:
      - No evidence of robust health monitoring, recovery, or auto-pull of models.
      - Non-Ollama providers (Anthropic, OpenAI, Gemini) have less robust life cycle management as well.
- **Remediation Plan**:
  - Develop an Ollama supervisor/healthcheck loop as a dedicated async task.
  - Implement auto-pulling and fallback logic for multiple models.
  - Emit clear error/status messages to the TUI if Ollama or other providers are unavailable or require setup.

---

## 4. Session Versioning, Migration, & Durability (Minor Caveat) ⚠️
- **README Claim**: Sessions are versioned and atomically migrated on load.
- **Finding**: Versioning and migration for session files claims to handle legacy session formats, but only basic JSON migration logic exists—complex migrations may require more robust handling in the future.
- **Remediation Plan**:
  - Implement explicit session file format versioning and automated migration with error reporting.
  - Test crash-safety claims and atomicity of writes/migrations under real-world situations.

---

## 5. Command/Tool Discovery, Dynamic Skill Loading (Partial) ⚠️
- **README Claim**: "Drop new Python/JS scripts into the `skills/` folder to instantly teach the agent new capabilities."
- **Finding**: Skill Executor supports this, but is limited to fixed scripts with naming conventions and permissions (e.g., `run.sh`, `run.py`, `run.js`). Security and discoverability limitations should be documented.
- **Remediation Plan**:
  - Expand documentation with file/folder naming, permissions, and discovered tool schema.
  - Consider adding plugin reload/scan shortcut for TUI.

---

## Additional General Findings
- Some docs overstate the readiness of "advanced context compaction/optimization," while most logic relies on underlying libraries (`mentalist`, `mindpalace`). Recommend clarifying which features are native to gypsy and which are inherited from dependencies.
- Periodic discrepancies may arise as dependencies evolve. Recommend explicit dependency version pinning for critical features and regular integration testing.

## General Remediation Suggestions
- Add a README matrix/grid stating each major feature claim and its status: ✅ (Available), ⚠️ (Partial), ⛔ (Missing/Roadmap)
- Open child issues for each major feature-gap and assign owners or milestones as needed.
- For any new or partial implementation, provide CLI and API usage examples in the docs.