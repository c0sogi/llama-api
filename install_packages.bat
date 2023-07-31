set VENV_DIR=.venv

if not exist %VENV_DIR% (
    echo Creating virtual environment
    python -m venv %VENV_DIR%
)
call %VENV_DIR%\Scripts\activate.bat
python -m llama_api.server.app_settings --install-pkgs