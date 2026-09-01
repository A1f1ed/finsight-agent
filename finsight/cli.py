"""FinSight Agent command-line interface (lightweight entry point without the web UI).

Run:
    python -m finsight.cli
"""

from langchain.messages import HumanMessage

from .agent import create_finsight_agent, new_thread_id


def main() -> None:
    thread_id = new_thread_id()
    print("Initializing FinSight (indexing knowledge base + preparing sandbox)...")
    agent, _ = create_finsight_agent(thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    print("\nFinSight ready. Type a question to start; type exit to quit.\n")
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
