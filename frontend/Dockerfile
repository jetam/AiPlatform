# Matches the current startStaticServer.sh exactly -- a plain static file
# server. Replacing this with nginx (and unifying it with the backend behind
# one port/reverse proxy) is Stage 6, intentionally not done here.
FROM python:3.10-slim

WORKDIR /app/frontend

COPY frontend/ /app/frontend/

EXPOSE 3000

CMD ["python3", "-m", "http.server", "3000"]
