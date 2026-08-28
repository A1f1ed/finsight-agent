"""冒烟测试：验证 RAG 知识问答链路（检索 -> 子代理分析 -> 汇总）。

运行:
    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.messages import HumanMessage

from finsight.agent import create_finsight_agent, new_thread_id


def main() -> None:
    thread_id = new_thread_id()
    agent, _ = create_finsight_agent(thread_id)

    question = "夏普比率怎么计算？请给出公式和解读标准。"
    print(f"Question: {question}\n")

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}},
    )
    print("Answer:")
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
