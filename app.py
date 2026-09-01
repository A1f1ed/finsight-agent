"""FinSight Agent — Streamlit chat interface.

Run:
    streamlit run app.py

(.streamlit/config.toml disables the file watcher so it doesn't scan unrelated deps)

UI features:
- Multi-turn conversation (LangGraph checkpointer keeps context per thread)
- Knowledge Q&A: finance concept answers with knowledge-base citations
- Data analysis: the agent generates charts in a local sandbox; the UI
  automatically displays the PNGs and reports
"""

import io
import logging
import os
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import streamlit as st

# The dependency chain indirectly imports transformers (which this project doesn't
# need); on import it prints a few docstring self-check logs for its own models
# (e.g. "[ERROR] ... paddleocr_vl ..."). Harmless but alarming: swallow stdout and
# silence its logger to CRITICAL.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

with redirect_stdout(io.StringIO()):
    from langchain.messages import HumanMessage

    from finsight.agent import OUTPUT_ROOT, SAMPLE_CSV, WORKSPACE_ROOT, create_finsight_agent, new_thread_id

st.set_page_config(page_title="FinSight Agent", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Session init: one thread_id + isolated workspace per conversation
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_dataset_preview() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


def init_session() -> None:
    """Initialize the agent and session state on first visit or 'New conversation'."""
    thread_id = new_thread_id()
    with st.spinner("Initializing FinSight (indexing knowledge base + preparing sandbox)..."):
        agent, backend = create_finsight_agent(thread_id)
    st.session_state.thread_id = thread_id
    st.session_state.agent = agent
    st.session_state.backend = backend
    st.session_state.messages = []


if "thread_id" not in st.session_state:
    init_session()

# ---------------------------------------------------------------------------
# Sidebar: project intro / dataset preview / suggested questions
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 FinSight Agent")
    st.caption(
        "A finance research assistant built on LangChain Deep Agents: "
        "RAG knowledge Q&A + sandboxed data analysis."
    )

    if st.button("🔄 Start new conversation", width="stretch"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Example questions")
    examples = [
        "How are the Sharpe ratio and maximum drawdown calculated?",
        "What are board lots and T+2 settlement in the HK stock market?",
        "Analyze the period returns and annualized volatility of the three stocks, and plot a comparison chart.",
        "Compute the maximum drawdown of each stock, and generate charts plus an analysis report.",
    ]
    for example in examples:
        if st.button(example, width="stretch"):
            st.session_state.pending_prompt = example

    st.subheader("Dataset preview")
    df = load_dataset_preview()
    st.caption(f"Simulated HK daily OHLCV · {df['Symbol'].nunique()} stocks · {len(df)} rows")
    st.dataframe(df.head(8), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Conversation area
# ---------------------------------------------------------------------------


def render_artifacts(artifact_paths: list[str]) -> None:
    """Render artifacts delivered by the agent: images inline, markdown reports expandable/downloadable.

    Download buttons must have an explicit unique key: different turns may deliver
    files with the same name and content (e.g. multiple analyses all called
    report.md), in which case Streamlit's auto-computed element IDs collide and it
    raises StreamlitDuplicateElementId. Keying on the full file path guarantees uniqueness.
    """
    for index, path_str in enumerate(dict.fromkeys(artifact_paths)):
        path = Path(path_str)
        if not path.exists():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            st.image(str(path), width="stretch")
        elif path.suffix.lower() == ".md":
            with st.expander(f"📄 View report: {path.name}"):
                st.markdown(path.read_text(encoding="utf-8"))
            st.download_button(
                f"⬇️ Download {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/markdown",
                key=f"dl-{index}-{path_str}",
            )


def collect_new_artifacts(before: set[str]) -> list[str]:
    """Diff the output tree before/after invoke to find this turn's new artifacts."""
    run_root = OUTPUT_ROOT / st.session_state.thread_id
    if not run_root.exists():
        return []
    current = {str(p) for p in run_root.rglob("*") if p.is_file()}
    return sorted(current - before)


def workspace_files() -> set[str]:
    """All files currently in the session workspace (including retrieved chunks)."""
    ws_root = WORKSPACE_ROOT / st.session_state.thread_id
    if not ws_root.exists():
        return set()
    return {str(p) for p in ws_root.rglob("*") if p.is_file()}


def extract_sources(new_files: set[str]) -> list[str]:
    """Parse citations from chunks newly retrieved this turn, deduped and order-preserving.

    Programmatic citations: when search_knowledge writes a chunk file, its first
    line is "# Source: knowledge/xxx.md". Parse that directly instead of relying on
    the LLM to include sources in its answer (which is non-deterministic).
    """
    sources: list[str] = []
    for path_str in sorted(new_files):
        path = Path(path_str)
        if path.suffix != ".md" or "retrieved" not in path.parts:
            continue
        try:
            first_line = path.read_text(encoding="utf-8").split("\n", 1)[0]
        except OSError:
            continue
        if first_line.startswith("# Source: "):
            src = first_line[len("# Source: "):].strip()
            if src and src not in sources:
                sources.append(src)
    return sources


def render_sources(sources: list[str]) -> None:
    """Render this turn's knowledge-base sources beneath the answer."""
    if sources:
        st.caption("📚 Sources: " + " · ".join(f"`{s}`" for s in sources))


st.header("FinSight · Finance Research AI Assistant")

# Render conversation history (including previously delivered charts/reports and sources)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("artifacts"):
            render_artifacts(msg["artifacts"])
        if msg.get("sources"):
            render_sources(msg["sources"])


prompt = st.chat_input("Ask a finance knowledge question, or have the assistant analyze the stock data…")

# Sidebar suggested questions: run directly as this turn's prompt
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    run_root = OUTPUT_ROOT / st.session_state.thread_id
    before = {str(p) for p in run_root.rglob("*") if p.is_file()} if run_root.exists() else set()
    ws_before = workspace_files()

    with st.chat_message("assistant"):
        with st.spinner(
            "FinSight is working (retrieving knowledge / writing and executing analysis code)… "
            "roughly 1-2 min locally; the cloud instance is overseas and calls China-based "
            "model endpoints, so please allow 3-5 min"
        ):
            try:
                result = st.session_state.agent.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    {"configurable": {"thread_id": st.session_state.thread_id}},
                )
                answer = result["messages"][-1].text
            except Exception as e:  # noqa: BLE001 - the chat UI needs a fallback error display
                answer = f"⚠️ Runtime error: {e}"

        st.markdown(answer)
        artifacts = collect_new_artifacts(before)
        render_artifacts(artifacts)
        sources = extract_sources(workspace_files() - ws_before)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "artifacts": artifacts, "sources": sources}
    )
