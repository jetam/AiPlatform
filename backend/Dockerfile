# Plain python:3.10-slim is enough -- torch's PyPI wheel bundles its own CUDA
# runtime (the nvidia-cu* packages pulled in via requirements.txt), so this
# image doesn't need an nvidia/cuda base. What IS required is a host with the
# NVIDIA driver + nvidia-container-toolkit installed, and the container run
# with GPU access (see docker-compose.yml). Without that, torch silently
# falls back to CPU instead of failing loudly -- verify with:
#   docker compose exec backend python3 -c "import torch; print(torch.cuda.is_available())"
FROM python:3.10-slim

WORKDIR /app/backend

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/ /app/backend/

EXPOSE 5000

# --host 0.0.0.0 is required inside a container (uvicorn's default of
# 127.0.0.1 would only be reachable from inside the container itself, not
# through the published port).
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]
