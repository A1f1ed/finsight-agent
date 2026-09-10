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
import shutil
import threading
import time
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
# Conversation management
#
# Each conversation owns a thread_id, an agent (its LangGraph InMemorySaver
# checkpointer IS the multi-turn memory) and an isolated workspace/output dir.
# All conversations live in st.session_state, so switching between them is
# instant and preserves each one's visible history AND its LLM context — the
# agent object is kept alive, never rebuilt. (State is per browser session; a
# full page refresh starts over, which matches the in-memory checkpointer.)
# ---------------------------------------------------------------------------

DEFAULT_TITLE = "New conversation"


@st.cache_data(show_spinner=False)
def load_dataset_preview() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


def conversations() -> dict:
    """Registry of all conversations: {thread_id: {title, created_at, messages, agent, backend}}."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    return st.session_state.conversations


def create_conversation() -> str:
    """Create a fresh conversation (new thread_id + agent + workspace) and switch to it."""
    thread_id = new_thread_id()
    with st.spinner("Starting a new FinSight conversation…"):
        agent, backend = create_finsight_agent(thread_id)
    conversations()[thread_id] = {
        "title": DEFAULT_TITLE,
        "created_at": time.time(),
        "messages": [],
        "agent": agent,
        "backend": backend,
        "status": "idle",  # "idle" | "running" — whether a background invoke is in flight
    }
    st.session_state.current_thread = thread_id
    return thread_id


def switch_conversation(thread_id: str) -> None:
    """Make an existing conversation the active one."""
    st.session_state.current_thread = thread_id


def delete_conversation(thread_id: str) -> None:
    """Drop a conversation and remove its on-disk workspace/output; reselect a neighbour."""
    convs = conversations()
    convs.pop(thread_id, None)
    for root in (WORKSPACE_ROOT / thread_id, OUTPUT_ROOT / thread_id):
        shutil.rmtree(root, ignore_errors=True)
    if st.session_state.get("current_thread") == thread_id:
        if convs:
            st.session_state.current_thread = max(convs, key=lambda t: convs[t]["created_at"])
        else:
            create_conversation()


def current() -> dict:
    """The active conversation dict."""
    return conversations()[st.session_state.current_thread]


def ordered_threads() -> list[str]:
    """All conversation thread_ids, most recently created first."""
    convs = conversations()
    return sorted(convs, key=lambda t: convs[t]["created_at"], reverse=True)


if "conversations" not in st.session_state:
    create_conversation()

# ---------------------------------------------------------------------------
# Sidebar: project intro / dataset preview / suggested questions
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 FinSight Agent")
    st.caption(
        "A finance research assistant built on LangChain Deep Agents: "
        "RAG knowledge Q&A + sandboxed data analysis."
    )

    if st.button("➕ New conversation", width="stretch", type="primary"):
        create_conversation()
        st.rerun()

    # --- Conversation switcher: click a title to switch, 🗑️ to delete --------
    st.subheader("Conversations")
    convs = conversations()
    current_thread = st.session_state.current_thread
    for thread_id in ordered_threads():
        conv = convs[thread_id]
        is_current = thread_id == current_thread
        # ⏳ = a background run is in flight; switching away does NOT interrupt it,
        # the answer is written back to its own thread.
        if conv["status"] == "running":
            icon = "⏳"
        elif is_current:
            icon = "🟢"
        else:
            icon = "💬"
        help_text = (
            "Running in the background…"
            if conv["status"] == "running"
            else f"{len(conv['messages'])} messages"
        )
        pick, drop = st.columns([6, 1])
        with pick:
            if st.button(
                f"{icon} {conv['title']}",
                key=f"switch-{thread_id}",
                width="stretch",
                help=help_text,
            ):
                if not is_current:
                    switch_conversation(thread_id)
                    st.rerun()
        with drop:
            if st.button("🗑️", key=f"delete-{thread_id}", help="Delete this conversation"):
                delete_conversation(thread_id)
                st.rerun()

    st.subheader("Example questions")
    examples = [
        "How are the Sharpe ratio and maximum drawdown calculated?",
        "What are board lots and T+2 settlement in the HK stock market?",
        "Analyze the period returns and annualized volatility of the three stocks, and plot a comparison chart.",
        "Compute the maximum drawdown of each stock, and generate charts plus an analysis report.",
    ]
    current_running = convs[current_thread]["status"] == "running"
    for example in examples:
        if st.button(example, width="stretch", disabled=current_running):
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


def collect_new_artifacts(thread_id: str, before: set[str]) -> list[str]:
    """Diff the output tree before/after invoke to find this turn's new artifacts."""
    run_root = OUTPUT_ROOT / thread_id
    if not run_root.exists():
        return []
    now_files = {str(p) for p in run_root.rglob("*") if p.is_file()}
    return sorted(now_files - before)


def workspace_files(thread_id: str) -> set[str]:
    """All files currently in the session workspace (including retrieved chunks)."""
    ws_root = WORKSPACE_ROOT / thread_id
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


# ---------------------------------------------------------------------------
# Background execution
#
# Streamlit reruns the whole script on every interaction and aborts whatever is
# mid-execution, so a synchronous agent.invoke() (minutes long) would be killed
# the instant the user switches or starts a conversation — the answer is lost.
# Fix: run invoke() on a daemon thread that writes its result straight back into
# THAT conversation's dict, located by the thread_id captured at launch (not by
# "current", which may have changed). The worker never touches st.* or
# st.session_state (there is no ScriptRunContext off the main thread); it only
# mutates the plain dict it is handed, which the main thread shares via
# session_state. A fragment then polls the active conversation until it lands.
# ---------------------------------------------------------------------------


def _run_agent_worker(
    conv: dict, thread_id: str, prompt: str, before: set[str], ws_before: set[str]
) -> None:
    """Background thread body: invoke the agent, write the assistant turn back into conv."""
    try:
        result = conv["agent"].invoke(
            {"messages": [HumanMessage(content=prompt)]},
            {"configurable": {"thread_id": thread_id}},
        )
        answer = result["messages"][-1].text
    except Exception as e:  # noqa: BLE001 - surface the error in-chat instead of killing the thread
        answer = f"⚠️ Runtime error: {e}"
    artifacts = collect_new_artifacts(thread_id, before)
    sources = extract_sources(workspace_files(thread_id) - ws_before)
    conv["messages"].append(
        {"role": "assistant", "content": answer, "artifacts": artifacts, "sources": sources}
    )
    conv["status"] = "idle"
    conv["just_finished"] = True  # tells the fragment to do one full rerun (re-enable the input)


def start_agent_run(conv: dict, thread_id: str, prompt: str) -> None:
    """Record the user turn, snapshot the workspace, and launch the worker thread."""
    if not conv["messages"]:  # first message -> title the conversation from the prompt
        conv["title"] = (prompt[:38] + "…") if len(prompt) > 38 else prompt
    conv["messages"].append({"role": "user", "content": prompt})

    run_root = OUTPUT_ROOT / thread_id
    before = {str(p) for p in run_root.rglob("*") if p.is_file()} if run_root.exists() else set()
    ws_before = workspace_files(thread_id)

    conv["status"] = "running"
    conv.pop("just_finished", None)
    threading.Thread(
        target=_run_agent_worker,
        args=(conv, thread_id, prompt, before, ws_before),
        daemon=True,
    ).start()


@st.fragment(run_every=1)
def poll_running() -> None:
    """Live 'working' indicator for the active conversation, refreshed once a second.

    run_every makes Streamlit re-execute ONLY this fragment every second (the sidebar
    and the message history are untouched) and — crucially — it manages the fragment
    rerun timing itself, so we never call st.rerun(scope="fragment") by hand. Doing
    that during a full-app run is illegal and raises StreamlitAPIException.

    When the worker thread flips status back to 'idle' it also sets just_finished; we
    then trigger ONE full rerun to render the landed answer and re-enable the chat
    input. When idle with nothing pending, this fragment is a cheap no-op.
    """
    conv = current()
    if conv["status"] == "running":
        st.info(
            "⏳ FinSight is working (retrieving knowledge / writing and running analysis code)… "
            "roughly 1-2 min locally, 3-5 min on the cloud instance. You can switch to or start "
            "another conversation — this run keeps going in the background."
        )
    elif conv.get("just_finished"):
        conv["just_finished"] = False
        st.rerun()  # full rerun: render the landed answer + re-enable the chat input


st.header("FinSight · Finance Research AI Assistant")

conv = current()
thread_id = st.session_state.current_thread
is_running = conv["status"] == "running"

prompt = st.chat_input(
    "Ask a finance knowledge question, or have the assistant analyze the stock data…",
    disabled=is_running,
)

# Sidebar suggested questions: run directly as this turn's prompt
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt and not is_running:
    start_agent_run(conv, thread_id, prompt)
    st.rerun()  # refresh the sidebar (title + ⏳) and re-render with the input disabled

# Render the active conversation's history (snapshot the list: a worker may append).
for msg in list(conv["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("artifacts"):
            render_artifacts(msg["artifacts"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# Live "working" note + completion refresh (fragment-local, polls once a second).
poll_running()
