@echo off
setlocal
cd /d "%~dp0"
echo AI Funscript Generator - setup (downloads Python + the GPU libraries, about 3 GB, once).
echo Needs Windows, an NVIDIA graphics card and an internet connection. Everything stays in this folder.
echo.
set "UV=%~dp0bin\uv.exe"
if not exist "%UV%" (
  where uv >nul 2>nul && (for /f "delims=" %%U in ('where uv') do set "UV=%%U") || (
    echo Getting uv (Python installer^) ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%~dp0bin'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex" || goto :fail
  )
)
where ffmpeg >nul 2>nul || if not exist "%~dp0bin\ffmpeg.exe" (
  echo ffmpeg not found - installing it with winget ...
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements || (
    echo Could not install ffmpeg automatically. Put ffmpeg.exe and ffprobe.exe into the "bin" folder and run setup again.
    goto :fail
  )
)
set "UV_CACHE_DIR=%~dp0.uv-cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0.python"
if not exist ".venv\Scripts\python.exe" "%UV%" venv .venv --python 3.12 --python-preference only-managed || goto :fail
set "PATH=%~dp0bin;%PATH%"
set "TORCH_HOME=%~dp0torch_home"
rem GPU libraries: newest build first; older cards (GTX 900 / 10-series, Quadro P...) need an older build
for %%C in (cu128 cu126 cu118) do (
  echo.
  echo Installing the GPU libraries ^(%%C^) ...
  "%UV%" pip install --python .venv\Scripts\python.exe --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%%C || goto :fail
  .venv\Scripts\python.exe -m funscript_ai.gpu_check
  if not errorlevel 1 goto :gpu_ok
  if errorlevel 2 (echo This build does not support the card - trying an older one.) else goto :nogpu
)
echo None of the GPU library builds runs on this graphics card.
goto :fail
:nogpu
echo The generator needs an NVIDIA graphics card with an up-to-date driver.
goto :fail
:gpu_ok
"%UV%" pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :fail
if not exist "models\thrust\thrust_v4.pt" (
  echo.
  echo Downloading the AI models ...
  powershell -NoProfile -Command "Invoke-WebRequest https://github.com/ZuperTheGod/ofs-ai-funscript-generator/releases/latest/download/models.zip -OutFile models.zip; Expand-Archive models.zip -DestinationPath . -Force; Remove-Item models.zip" || goto :fail
)
echo.
echo Setup finished.
echo  - Drag videos onto "2 - Make funscripts.bat", or
echo  - run "3 - Install OFS extension.bat" to generate from inside OpenFunscripter.
pause
exit /b 0
:fail
echo.
echo Setup FAILED - please open an issue on GitHub with a screenshot of this window.
pause
exit /b 1
