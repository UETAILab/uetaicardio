#!/bin/bash

export PYTHONPATH="."
export DEBUG=1
export ENV="prod"

echo "-------------------------------------------"
echo "Starting worker"
python server/worker.py &

echo "Starting server"
python server/server.py
echo "-------------------------------------------"
