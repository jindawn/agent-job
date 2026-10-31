FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY agenthub ./agenthub
COPY data ./data
RUN pip install --no-cache-dir ".[server]" && useradd -m app && chown -R app /app
USER app
ENV DB_PATH=/app/data/demo.db
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "agenthub.server:app", "--host", "0.0.0.0", "--port", "8000"]
