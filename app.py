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
import json
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

    from finsight.agent import (
        OUTPUT_ROOT,
        SAMPLE_CSV,
        WORKSPACE_ROOT,
        create_finsight_agent,
        get_checkpointer,
        new_thread_id,
    )

st.set_page_config(page_title="FinSight Agent", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Conversation management
#
# Each conversation owns a thread_id, an agent and an isolated workspace/output dir.
# Multi-turn memory is the LangGraph SqliteSaver checkpointer (persistent, keyed by
# thread_id), so context survives a restart; the UI-side history is mirrored to a
# conversation.json sidecar and rebuilt on a fresh session, so a page refresh
# RESTORES your conversations instead of starting over. Within a live session all
# conversations stay in st.session_state, so switching is instant and never rebuilds.
# ---------------------------------------------------------------------------

DEFAULT_TITLE = "New conversation"

# --- Chat presentation -------------------------------------------------------
ASSISTANT_AVATAR = ":material/monitoring:"
USER_AVATAR = ":material/person:"
STREAM_CURSOR = "▌"  # trailing block shown while the answer is still streaming in

# Empty-state onboarding suggestions: short pill label -> full prompt.
SUGGESTIONS = {
    ":material/calculate: Sharpe & drawdown": (
        "How are the Sharpe ratio and maximum drawdown calculated?"
    ),
    ":material/schedule: Board lots & T+2": (
        "What are board lots and T+2 settlement in the HK stock market?"
    ),
    ":material/query_stats: Returns & volatility": (
        "Analyze the period returns and annualized volatility of the three stocks, "
        "and plot a comparison chart."
    ),
    ":material/description: Drawdown report": (
        "Compute the maximum drawdown of each stock, and generate charts plus an "
        "analysis report."
    ),
}


def avatar_for(role: str) -> str:
    """Material-icon avatar per chat role (assistant = finance chart, user = person)."""
    return ASSISTANT_AVATAR if role == "assistant" else USER_AVATAR


@st.cache_data(show_spinner=False)
def load_dataset_preview() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


def conversations() -> dict:
    """Registry of all conversations: {thread_id: {title, created_at, messages, agent, backend}}."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    return st.session_state.conversations


def save_conversation(conv: dict) -> None:
    """Persist a conversation's UI metadata + visible history to its workspace dir.

    The SQLite checkpointer already persists the model-side state per thread_id; this
    sidecar JSON persists what the *UI* needs to rebuild the sidebar and chat history
    after a refresh/restart (title, order, rendered messages with artifacts/sources).
    It lives inside the thread's workspace dir, so deleting the conversation removes
    it too. Plain file I/O — safe to call from the background worker thread.
    """
    thread_id = conv["thread_id"]
    meta = {
        "thread_id": thread_id,
        "title": conv["title"],
        "created_at": conv["created_at"],
        "messages": conv["messages"],
    }
    path = WORKSPACE_ROOT / thread_id / "conversation.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass  # best-effort: never break the UI over a persistence write failure


def create_conversation() -> str:
    """Create a fresh conversation (new thread_id + agent + workspace) and switch to it."""
    thread_id = new_thread_id()
    with st.spinner("Starting a new FinSight conversation…"):
        agent, backend = create_finsight_agent(thread_id)
    conversations()[thread_id] = {
        "thread_id": thread_id,
        "title": DEFAULT_TITLE,
        "created_at": time.time(),
        "messages": [],
        "agent": agent,
        "backend": backend,
        "status": "idle",  # "idle" | "running" — whether a background invoke is in flight
        "cancel": threading.Event(),  # cooperative stop signal checked by the worker between steps
        "stream_buffer": [],  # live token deltas for the in-flight answer (rendered by poll_running)
    }
    st.session_state.current_thread = thread_id
    save_conversation(conversations()[thread_id])
    return thread_id


def switch_conversation(thread_id: str) -> None:
    """Make an existing conversation the active one."""
    st.session_state.current_thread = thread_id


def delete_conversation(thread_id: str) -> None:
    """Drop a conversation, remove its workspace/output + persisted checkpoints; reselect a neighbour."""
    convs = conversations()
    convs.pop(thread_id, None)
    for root in (WORKSPACE_ROOT / thread_id, OUTPUT_ROOT / thread_id):
        shutil.rmtree(root, ignore_errors=True)  # also removes conversation.json
    try:
        get_checkpointer().delete_thread(thread_id)  # drop this thread's checkpoint rows
    except Exception:  # noqa: BLE001 - saver API may not expose delete_thread; non-fatal
        pass
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


def restore_conversations() -> None:
    """Rebuild the conversation registry from disk on a fresh session.

    Scans the workspace dir for persisted conversation.json sidecars, recreates each
    agent (rebinding to its thread_id, so the SQLite checkpointer restores the
    model-side context) and reloads the visible history. If nothing is found (first
    ever run), starts one new conversation. Idempotent within a session.
    """
    convs = conversations()
    if convs:  # already restored/created earlier this session
        return
    found: list[tuple[str, dict]] = []
    if WORKSPACE_ROOT.exists():
        for meta_path in sorted(WORKSPACE_ROOT.glob("*/conversation.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            found.append((meta.get("thread_id") or meta_path.parent.name, meta))
    if not found:
        create_conversation()
        return
    with st.spinner("Restoring your conversations…"):
        for tid, meta in found:
            agent, backend = create_finsight_agent(tid)
            convs[tid] = {
                "thread_id": tid,
                "title": meta.get("title", DEFAULT_TITLE),
                "created_at": meta.get("created_at", time.time()),
                "messages": meta.get("messages", []),
                "agent": agent,
                "backend": backend,
                "status": "idle",
                "cancel": threading.Event(),
                "stream_buffer": [],
            }
    st.session_state.current_thread = max(convs, key=lambda t: convs[t]["created_at"])


restore_conversations()

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
    """Background thread body: stream the agent (cancellable via conv["cancel"]), write the assistant turn back into conv."""
    cancel = conv["cancel"]
    buffer = conv["stream_buffer"]  # list[str]; the fragment renders "".join(buffer) live
    stopped = False
    collected = None
    try:
        # Two stream modes at once:
        #  - "messages": (AIMessageChunk, metadata) token deltas from the main graph's model
        #    node -> appended to the buffer so the answer types out live in the UI. (Subagent
        #    subgraphs are excluded by default; tool/thinking deltas arrive with empty content.)
        #  - "values": the full state after each step, whose last message is the authoritative
        #    final answer and our checkpoint to honour Stop between steps (invoke() can't).
        stream = conv["agent"].stream(
            {"messages": [HumanMessage(content=prompt)]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode=["messages", "values"],
        )
        try:
            for mode, payload in stream:
                if cancel.is_set():  # user clicked Stop -> halt at the next chunk/step boundary
                    stopped = True
                    break
                if mode == "messages":
                    chunk, _meta = payload
                    delta = getattr(chunk, "content", "")
                    if isinstance(delta, str) and delta:
                        buffer.append(delta)  # list.append is atomic under the GIL
                else:  # "values"
                    collected = payload
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()  # best-effort: stop the graph from advancing further
        if stopped:
            partial = "".join(buffer).strip()
            answer = "⏹ Stopped by user." + (f"\n\n{partial}" if partial else "")
        elif collected and collected.get("messages"):
            answer = collected["messages"][-1].text
        else:
            answer = "".join(buffer).strip() or "(no output)"
    except Exception as e:  # noqa: BLE001 - surface the error in-chat instead of killing the thread
        answer = f"⚠️ Runtime error: {e}"
    artifacts = collect_new_artifacts(thread_id, before)
    sources = extract_sources(workspace_files(thread_id) - ws_before)
    conv["messages"].append(
        {"role": "assistant", "content": answer, "artifacts": artifacts, "sources": sources}
    )
    conv["stream_buffer"] = []  # drop the live buffer; the landed message now renders from history
    conv["status"] = "idle"
    conv["just_finished"] = True  # tells the fragment to do one full rerun (re-enable the input)
    save_conversation(conv)  # persist the landed assistant turn so it survives a refresh


def start_agent_run(conv: dict, thread_id: str, prompt: str) -> None:
    """Record the user turn, snapshot the workspace, and launch the worker thread."""
    if not conv["messages"]:  # first message -> title the conversation from the prompt
        conv["title"] = (prompt[:38] + "…") if len(prompt) > 38 else prompt
    conv["messages"].append({"role": "user", "content": prompt})
    save_conversation(conv)  # persist the user turn + (new) title immediately

    run_root = OUTPUT_ROOT / thread_id
    before = {str(p) for p in run_root.rglob("*") if p.is_file()} if run_root.exists() else set()
    ws_before = workspace_files(thread_id)

    conv["status"] = "running"
    conv["cancel"].clear()  # reset the stop signal for this fresh run
    conv["stream_buffer"] = []  # reset the live streaming buffer for this fresh run
    conv.pop("just_finished", None)
    threading.Thread(
        target=_run_agent_worker,
        args=(conv, thread_id, prompt, before, ws_before),
        daemon=True,
    ).start()


@st.fragment(run_every=0.4)
def poll_running() -> None:
    """Live streaming surface for the active conversation, refreshed ~2.5x/sec.

    While a run is in flight the background worker appends token deltas to
    conv["stream_buffer"]; this fragment renders them as a growing assistant message so
    the answer types out live. run_every re-executes ONLY this fragment (the sidebar and
    the message history are untouched) and — crucially — Streamlit manages the fragment
    rerun timing itself, so we never call st.rerun(scope="fragment") by hand (illegal
    during a full-app run; raises StreamlitAPIException).

    When the worker lands the final message it flips status to 'idle' + just_finished; we
    then trigger ONE full rerun to render the polished answer (with artifacts/sources) and
    re-enable the chat input. When idle with nothing pending, this fragment is a cheap no-op.
    """
    conv = current()
    if conv["status"] != "running":
        if conv.get("just_finished"):
            conv["just_finished"] = False
            st.rerun()  # full rerun: render the landed answer + re-enable the chat input
        return

    stopping = conv["cancel"].is_set()
    text = "".join(conv.get("stream_buffer") or [])
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        if text.strip():
            st.markdown(text + STREAM_CURSOR)  # answer streaming in, with a live cursor
        elif stopping:
            st.caption(":material/stop_circle: Stopping… the current step finishes, then the run halts.")
        else:
            st.caption(
                ":material/psychology: FinSight is thinking — retrieving knowledge / writing "
                "and running analysis code. You can switch conversations; this keeps running."
            )

    # Stop lives in this fragment: clicking it reruns only the fragment and sets the
    # cooperative cancel flag, so the worker halts at its next chunk/step boundary.
    if not stopping and st.button(
        "Stop generating", icon=":material/stop_circle:", key="stop-run"
    ):
        conv["cancel"].set()


st.header("FinSight · Finance Research AI Assistant")

conv = current()
thread_id = st.session_state.current_thread
is_running = conv["status"] == "running"

prompt = st.chat_input(
    "Ask a finance knowledge question, or have the assistant analyze the stock data…",
    disabled=is_running,
)

# Sidebar / welcome suggestions: run directly as this turn's prompt
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt and not is_running:
    start_agent_run(conv, thread_id, prompt)
    st.rerun()  # refresh the sidebar (title + ⏳) and re-render with the input disabled

# Empty state: a short welcome + one-click suggestion pills (disappear once chatting).
if not conv["messages"]:
    st.markdown(
        "#### :material/monitoring: What can I help you with?\n"
        "Ask a finance **knowledge** question, or have me **analyze the sample HK stock "
        "data** — returns, volatility, drawdown — with charts and a report."
    )
    picked = st.pills(
        "Try one of these",
        list(SUGGESTIONS),
        key=f"suggest-{thread_id}",
        label_visibility="collapsed",
        disabled=is_running,
    )
    if picked:
        st.session_state.pending_prompt = SUGGESTIONS[picked]
        st.rerun()

# Render the active conversation's history (snapshot the list: a worker may append).
for msg in list(conv["messages"]):
    with st.chat_message(msg["role"], avatar=avatar_for(msg["role"])):
        st.markdown(msg["content"])
        if msg.get("artifacts"):
            render_artifacts(msg["artifacts"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# Live streaming answer + Stop control + completion refresh (fragment-local).
poll_running()
