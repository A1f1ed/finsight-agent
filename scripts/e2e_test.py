"""End-to-end test: data-analysis workflow (write code -> execute -> chart -> report delivery)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.messages import HumanMessage

from finsight.agent import OUTPUT_ROOT, create_finsight_agent, new_thread_id


def main() -> None:
    thread_id = new_thread_id()
    agent, _ = create_finsight_agent(thread_id)

    question = "Compute the period return and annualized volatility of each of the three stocks, plot a comparison chart, and generate a short analysis report."
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
