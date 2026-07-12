#!/bin/sh

sudo chown -R $(whoami):$(whoami) /home/$(whoami)/app/.venv
sudo chown -R $(whoami):$(whoami) ~/.cache
uv sync
