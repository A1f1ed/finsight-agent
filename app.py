"""FinSight Agent — Streamlit 对话界面。

运行:
    streamlit run app.py

（.streamlit/config.toml 已关闭文件监视，避免它扫描无关依赖）

界面功能：
- 多轮对话（基于 LangGraph checkpointer 记住上下文）
- 知识问答：带知识库引用的金融概念解答
- 数据分析：agent 在本地沙箱生成图表，界面自动展示 PNG 与报告
"""

import io
import logging
import os
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import streamlit as st

# 依赖链会间接导入 transformers（本项目并不需要它），其内部在导入时会用 print
# 输出几条自家模型的 docstring 自检日志（形如 [ERROR] ... paddleocr_vl ...）。
# 这些日志无害但很吓人：用空 stdout 拦截，并把它的 logger 压到 CRITICAL。
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

with redirect_stdout(io.StringIO()):
    from langchain.messages import HumanMessage

    from finsight.agent import OUTPUT_ROOT, SAMPLE_CSV, create_finsight_agent, new_thread_id

st.set_page_config(page_title="FinSight Agent", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# 会话状态初始化：每个会话一个 thread_id + 独立工作区
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_dataset_preview() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


def init_session() -> None:
    """首次进入或点击"新对话"时初始化 agent 与会话状态。"""
    thread_id = new_thread_id()
    with st.spinner("正在初始化 FinSight（索引知识库 + 准备沙箱）..."):
        agent, backend = create_finsight_agent(thread_id)
    st.session_state.thread_id = thread_id
    st.session_state.agent = agent
    st.session_state.backend = backend
    st.session_state.messages = []


if "thread_id" not in st.session_state:
    init_session()

# ---------------------------------------------------------------------------
# 侧边栏：项目介绍 / 数据集预览 / 推荐问题
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 FinSight Agent")
    st.caption(
        "基于 LangChain Deep Agents 的金融研究助手："
        "RAG 知识问答 + 沙箱数据分析。"
    )

    if st.button("🔄 开始新对话", width="stretch"):
        st.session_state.clear()
        st.rerun()

    st.subheader("示例问题")
    examples = [
        "夏普比率和最大回撤分别怎么计算？",
        "港股的每手股数和 T+2 交收是怎么回事？",
        "分析三只股票的区间涨跌幅和年化波动率，并画一张对比图。",
        "计算每只股票的最大回撤，生成图表和一份分析报告。",
    ]
    for example in examples:
        if st.button(example, width="stretch"):
            st.session_state.pending_prompt = example

    st.subheader("数据集预览")
    df = load_dataset_preview()
    st.caption(f"模拟港股日线行情 · {df['Symbol'].nunique()} 只股票 · {len(df)} 行")
    st.dataframe(df.head(8), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 对话区
# ---------------------------------------------------------------------------


def render_artifacts(artifact_paths: list[str]) -> None:
    """渲染 agent 交付的产物：图片直接展示，Markdown 报告可展开/下载。"""
    for path_str in artifact_paths:
        path = Path(path_str)
        if not path.exists():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            st.image(str(path), width="stretch")
        elif path.suffix.lower() == ".md":
            with st.expander(f"📄 查看报告：{path.name}"):
                st.markdown(path.read_text(encoding="utf-8"))
            st.download_button(
                f"⬇️ 下载 {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/markdown",
            )


def collect_new_artifacts(before: set[str]) -> list[str]:
    """对比 invoke 前后，找出本轮新产出的交付物。"""
    run_root = OUTPUT_ROOT / st.session_state.thread_id
    if not run_root.exists():
        return []
    current = {str(p) for p in run_root.rglob("*") if p.is_file()}
    return sorted(current - before)


st.header("FinSight · 金融研究 AI 助手")

# 渲染历史消息（含历史交付的图表/报告）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("artifacts"):
            render_artifacts(msg["artifacts"])


prompt = st.chat_input("问一个金融知识问题，或让助手分析股票数据…")

# 侧边栏推荐问题填入输入框流程：直接作为本轮提问执行
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    run_root = OUTPUT_ROOT / st.session_state.thread_id
    before = {str(p) for p in run_root.rglob("*") if p.is_file()} if run_root.exists() else set()

    with st.chat_message("assistant"):
        with st.spinner("FinSight 正在工作（检索知识 / 编写并执行分析代码，可能需要 1-2 分钟）…"):
            try:
                result = st.session_state.agent.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    {"configurable": {"thread_id": st.session_state.thread_id}},
                )
                answer = result["messages"][-1].text
            except Exception as e:  # noqa: BLE001 - 对话界面需要兜底错误展示
                answer = f"⚠️ 运行出错：{e}"

        st.markdown(answer)
        artifacts = collect_new_artifacts(before)
        render_artifacts(artifacts)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "artifacts": artifacts}
    )
