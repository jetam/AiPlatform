# Matches startStaticServer.sh
FROM python:3.10-slim

WORKDIR /app/frontend

COPY frontend/ /app/frontend/

EXPOSE 3000

CMD ["python3", "-m", "http.server", "3000"]
