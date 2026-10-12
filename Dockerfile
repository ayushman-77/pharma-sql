# API + UI image. The LLM runs in Ollama outside this container:
#   docker build -t pharmasql .
#   docker run -p 8000:8000 -e PHARMASQL_OLLAMA_BASE_URL=http://host.docker.internal:11434 pharmasql
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY config ./config
COPY app ./app
COPY data ./data
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501
CMD ["pharmasql", "serve", "--host", "0.0.0.0", "--port", "8000"]
