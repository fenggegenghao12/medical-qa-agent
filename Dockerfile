# 后端镜像：Python 3.11 slim（与本地 venv 版本一致，保证 bm25.pkl 兼容）
FROM python:3.11-slim
WORKDIR /app

# 模型缓存已随项目打包在 ./huggingface/hub 下，HF_HOME 指过去即可离线命中
ENV HF_HOME=/app/huggingface

# 先装 CPU 版 torch：避免 pip 默认拉取 2GB+ 的 CUDA 版（本项目纯 CPU 推理）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# 阿里云源 + 超时重试：清华源偶发拉空索引页（from versions: none）时更稳
RUN pip install --no-cache-dir --default-timeout=100 --retries 5 \
    -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . .

# 启动：监听所有网卡（容器内必须 0.0.0.0，否则宿主机访问不到）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
