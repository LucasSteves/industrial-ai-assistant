@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Ambiente nao encontrado. Execute primeiro 01_INSTALAR.bat
    pause
    exit /b 1
)

rem O PATH pode nao enxergar um Ollama recem-instalado; por isso tambem procura no local padrao.
set "OLLAMA="
where ollama >nul 2>&1 && set "OLLAMA=ollama"
if not defined OLLAMA if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set OLLAMA="%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLLAMA (
    echo ERRO: Ollama nao encontrado. Execute primeiro 01_INSTALAR.bat
    pause
    exit /b 1
)

powershell -NoProfile -Command "try { Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo Iniciando o Ollama em segundo plano...
    start "" /min %OLLAMA% serve
    timeout /t 5 /nobreak >nul
)

echo Abrindo o Industrial AI Assistant no navegador...
echo Para encerrar, feche esta janela.
".venv\Scripts\python.exe" -m streamlit run app.py
