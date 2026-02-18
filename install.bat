@echo off
REM ============================================================================
REM Talking Head Video Generator — Windows Installer
REM One-click venv creation + dependency install
REM ============================================================================

echo ============================================================
echo   Talking Head Video Generator — Windows Installer
echo ============================================================
echo.

REM Run the Python install script
python install.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Installation failed. See output above.
    pause
    exit /b 1
)

echo.
echo Installation complete. To activate the environment:
echo   .\venv\Scripts\activate
echo.
pause
