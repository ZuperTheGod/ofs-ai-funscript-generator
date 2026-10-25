@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run "1 - Setup (run once).bat" first.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Drag one or more videos, or a folder, onto this file.
  echo A funscript is written next to each video - or ^<video^>.ai.funscript if the video already has one
  echo ^(your own scripts are never overwritten^).
  pause
  exit /b 1
)
set "PATH=%~dp0bin;%PATH%"
set "TORCH_HOME=%~dp0torch_home"
.venv\Scripts\python.exe -m funscript_ai.generate %*
pause
