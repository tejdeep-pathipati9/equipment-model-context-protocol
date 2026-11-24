#!/bin/sh
# Seed the DB if missing (keeps escalations on a mounted /app/data), then run the agent.
set -e
seed
exec agent "$@"
