# Matches startStaticServer.sh -- that script has no `cd`, so it serves from
# the PROJECT ROOT (paths like /frontend/composer.html). WORKDIR here is /app
# (not /app/frontend) so the container reproduces that same serving root
# instead of serving from inside frontend/ itself.
FROM python:3.10-slim

WORKDIR /app

COPY frontend/ /app/frontend/

EXPOSE 3000

CMD ["python3", "-m", "http.server", "3000"]
