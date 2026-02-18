FROM python:3.12-slim

WORKDIR /app

# Install Python deps (torch CPU-only, then rest from PyPI)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir 'fastapi[standard]' numpy

# Copy app code + model checkpoint
COPY _01_simulator/ _01_simulator/
COPY _02_agents/ _02_agents/
COPY _04_ui/ _04_ui/
COPY checkpoints/ checkpoints/
COPY __init__.py .
COPY pyproject.toml .

# Expose port (HuggingFace Spaces uses 7860)
EXPOSE 7860

# Run
ENV NEURAL_CHECKPOINT=checkpoints/model_inference.pt
CMD ["python", "-m", "uvicorn", "_04_ui.app:app", "--host", "0.0.0.0", "--port", "7860"]
