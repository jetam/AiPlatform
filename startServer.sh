#!/usr/bin/env bash

# Dynamic api server. starts server.py, trains neural network

cd "$(dirname "$0")/backend" || exit 1

kill $(lsof -t -i:5000)


> ../server_api.log


DYNAMIC_PORT=5000
echo "Starting api server on port $DYNAMIC_PORT"
PYTHONUNBUFFERED=1 python3 -m uvicorn server:app --port 5000 | tee ../server_api.log  #PYTHONUNBUFFERED=1  -> does not buffer output. logs dont work without this
