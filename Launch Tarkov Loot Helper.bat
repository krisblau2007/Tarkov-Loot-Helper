@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating virtual environment...
    python -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || goto :error
)

if not exist "data\icons" (
    echo Downloading item data and icons ^(one-time, may take a few minutes^)...
    ".venv\Scripts\python.exe" -m tarkov_loot_helper setup || goto :error
)

echo Starting Tarkov Loot Helper - look for the tray icon. Close this window to quit.
".venv\Scripts\python.exe" -m tarkov_loot_helper || goto :error
exit /b 0

:error
echo.
echo ============================================================
echo  Tarkov Loot Helper failed. Read the messages above.
echo ============================================================
set /p "_wait=Press Enter to close this window..."
exit /b 1
