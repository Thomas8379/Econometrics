@echo off
setlocal
REM econtools CLI launcher
python -m econtools %*

REM If launched via double-click (cmd /c), keep the window open.
echo %cmdcmdline% | find /i "/c" >nul
if not errorlevel 1 (
  echo.
  pause
)
