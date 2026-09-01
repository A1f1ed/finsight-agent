# FinSight Agent — Docker image
#
# Build:  docker build -t finsight-agent .
# Run:    docker run --env-file .env -p 8000:8000 finsight-agent
# Notes:  --env-file .env injects the local .env into the container at runtime
#         (secrets never go into the image); the container runs the FastAPI
#         service (finsight/server.py), and sandboxed code execution also happens
#         inside the container, so pandas/matplotlib are included in the image.

FROM python:3.12-slim

WORKDIR /app

# Copy the dependency manifest and install first to leverage Docker layer caching:
# dependencies are not reinstalled when only code changes, keeping builds fast
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the project code (.dockerignore excludes .env, runtime artifacts, etc.)
COPY . .

EXPOSE 8000

# Sync endpoints wrap long-running tasks, so a single process can serve multiple
# concurrent sessions (FastAPI dispatches sync endpoints to a thread pool)
CMD ["uvicorn", "finsight.server:app", "--host", "0.0.0.0", "--port", "8000"]
