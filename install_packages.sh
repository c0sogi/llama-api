#!/bin/bash
VENV_DIR=.venv

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment"
    python3 -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate
python3 -m llama_api.server.app_settings --install-pkgs