@echo off
REM One-click hot restart for the novel pipeline (Windows).
REM Kills any running run.py / pipeline.py for this project and relaunches it detached.
REM The pipeline resumes from the last unfinished chapter via existing checkpoints.

setlocal
cd /d "%~dp0"

REM Prefer the project venv (LibreOffice's bundled python lacks `openai`).
set "VENV_PY=E:\pycharmproject\allvenv\novel\Scripts\python.exe"
if exist "%VENV_PY%" (
    "%VENV_PY%" restart.py %*
) else (
    REM Fall back to whatever python is on PATH; user must ensure it has deps.
    python restart.py %*
)

endlocal
