"""端到端测试：数据分析工作流（写代码 -> 执行 -> 出图 -> 交付报告）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.messages import HumanMessage

from finsight.agent import OUTPUT_ROOT, create_finsight_agent, new_thread_id


def main() -> None:
    thread_id = new_thread_id()
    agent, _ = create_finsight_agent(thread_id)

    question = "计算三只股票各自的区间涨跌幅和年化波动率，画一张对比图，并生成一份简短分析报告。"
    print(f"Question: {question}\n")

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}},
    )
    print("=" * 60)
    print("Final answer:")
    print(result["messages"][-1].text)

    print("\n" + "=" * 60)
    print("Artifacts delivered to:", OUTPUT_ROOT / thread_id)
    for p in sorted((OUTPUT_ROOT / thread_id).rglob("*")):
        if p.is_file():
            print(f"  {p.name}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
