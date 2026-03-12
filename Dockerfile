FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates nodejs npm && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    gymnasium>=0.29.0 numpy>=1.24.0 \
    stable-baselines3>=2.1.0 pytest>=7.0

COPY start_agent.sh /usr/local/bin/start_agent.sh
RUN chmod +x /usr/local/bin/start_agent.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/start_agent.sh"]
