@echo off
REM === Tactical Trading Bot - double-click launcher (Windows) ===
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
where py >nul 2>nul && (set PY=py) || (set PY=python)
if not exist ".venv" (
  echo First run: setting up the environment, one moment...
  %PY% -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
python menu.py
pause
