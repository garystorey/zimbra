@echo off
REM ============================================================================
REM Video Generator— Run Pipeline
REM Activates the venv and passes all arguments to generate.py
REM ============================================================================


    
if not exist ".\venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Run install.bat first.
    echo.
    pause
    exit /b 1
)

REM Run the pipeline, forwarding all command-line arguments
.\venv\Scripts\python.exe generate.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Generation failed. See output above.
    pause
    exit /b 1
)

echo.
pause
