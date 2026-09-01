"""API smoke test: calls the FastAPI service's /chat endpoint.

Start the service first:
    uvicorn finsight.server:app --host 127.0.0.1 --port 8000

Then run:
    python scripts/api_test.py
"""

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def post_chat(message: str, thread_id: str | None = None) -> dict:
    payload = {"message": message}
    if thread_id:
        payload["thread_id"] = thread_id
    req = urllib.request.Request(
        f"{BASE}/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    question = "What is maximum drawdown and how is it calculated?"
    print(f"POST /chat  message={question!r}")

    start = time.time()
    result = post_chat(question)
    print(f"done in {time.time() - start:.0f}s")
    print("thread_id:", result["thread_id"])
    print("artifacts:", result["artifacts"])
    print("answer preview:")
    print(result["answer"][:400])


if __name__ == "__main__":
    main()
