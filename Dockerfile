# FinSight Agent — Docker 镜像
#
# 构建:  docker build -t finsight-agent .
# 运行:  docker run --env-file .env -p 8000:8000 finsight-agent
# 说明:  --env-file .env 把本地 .env 注入容器（密钥不进镜像）；
#        容器内运行的是 FastAPI 服务（finsight/server.py），
#        沙箱代码执行也在容器内完成，因此镜像里包含 pandas/matplotlib。

FROM python:3.12-slim

WORKDIR /app

# 先复制依赖清单并安装，利用 Docker 层缓存：
# 代码改动时不会重新安装依赖，构建更快
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制项目代码（.dockerignore 已排除 .env、运行时产物等）
COPY . .

EXPOSE 8000

# 同步端点内部是长耗时任务，单进程即可承载多个并发会话
# （FastAPI 会把同步端点派发到线程池）
CMD ["uvicorn", "finsight.server:app", "--host", "0.0.0.0", "--port", "8000"]
