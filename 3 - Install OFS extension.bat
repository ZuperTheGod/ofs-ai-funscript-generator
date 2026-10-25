@echo off
setlocal
set "DST=%APPDATA%\OFS\OFS3_data\extensions\AI Funscript Generator"
if not exist "%DST%" mkdir "%DST%"
copy /y "%~dp0ofs-extension\AI Funscript Generator\main.lua" "%DST%\main.lua" >nul || goto :fail
set "GEN=%~dp0"
if "%GEN:~-1%"=="\" set "GEN=%GEN:~0,-1%"
>"%DST%\settings.txt" echo generatorDir=%GEN%
echo Installed into %DST%
echo In OpenFunscripter: Extensions menu - enable "AI Funscript Generator" - Show window.
pause
exit /b 0
:fail
echo Could not install - is OpenFunscripter 3 installed (it creates %APPDATA%\OFS\OFS3_data)?
pause
exit /b 1
