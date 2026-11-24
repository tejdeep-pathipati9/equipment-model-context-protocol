#!/bin/sh
# `server`: seed the DB if missing, then serve MCP. Anything else: run the agent with those arguments.
set -e
if [ "$1" = "server" ]; then
    seed
    exec server
fi
exec agent "$@"
