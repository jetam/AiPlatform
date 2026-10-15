
FROM python:3.10-slim

WORKDIR /app/backend

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/ /app/backend/

EXPOSE 5000

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]
