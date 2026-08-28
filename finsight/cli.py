"""FinSight Agent 命令行界面（不想开网页时的轻量入口）。

运行:
    python -m finsight.cli
"""

from langchain.messages import HumanMessage

from .agent import create_finsight_agent, new_thread_id


def main() -> None:
    thread_id = new_thread_id()
    print("正在初始化 FinSight（索引知识库 + 准备沙箱）...")
    agent, _ = create_finsight_agent(thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    print("\nFinSight 就绪。输入问题开始对话，输入 exit 退出。\n")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]}, config
        )
        print(f"\nFinSight> {result['messages'][-1].text}\n")


if __name__ == "__main__":
    main()
