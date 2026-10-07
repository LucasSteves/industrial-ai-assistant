@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo   INDUSTRIAL AI ASSISTANT - INSTALACAO
echo   Rocket Industrial Solutions (RAG)
echo ==========================================================
echo.
echo Este instalador baixa e instala tudo o que falta:
echo   1. Python 3.12 (se nao houver Python 3.10 a 3.12)
echo   2. Bibliotecas do projeto (ambiente .venv)
echo   3. Ollama + modelo llama3.2 (cerca de 2 GB)
echo   4. Modelo de embeddings e indexacao da pasta knowledge
echo.

rem ---------------------------------------------------------------- 1. Python
echo [1/4] Procurando Python 3.10 a 3.12...
call :find_python
if defined PY goto python_ok

echo Python compativel nao encontrado. Instalando Python 3.12 pelo winget...
where winget >nul 2>&1
if errorlevel 1 goto no_winget
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
call :find_python
if defined PY goto python_ok

:no_winget
echo.
echo ERRO: nao foi possivel instalar o Python automaticamente.
echo Instale o Python 3.12 em https://www.python.org/downloads/
echo marcando "Add Python to PATH" e rode este instalador novamente.
echo Versoes mais novas (3.13, 3.14) nao sao suportadas pelas bibliotecas do projeto.
pause
exit /b 1

:python_ok
echo Usando: %PY%
%PY% --version

rem ------------------------------------------------------------ 2. bibliotecas
echo.
echo [2/4] Instalando bibliotecas (pode demorar alguns minutos na primeira vez)...
if not exist ".venv\Scripts\python.exe" (
    echo Criando ambiente virtual .venv ...
    %PY% -m venv .venv || (echo Falha ao criar o ambiente virtual. & pause & exit /b 1)
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERRO na instalacao das bibliotecas. Verifique a internet e rode novamente.
    pause
    exit /b 1
)

rem ---------------------------------------------------------------- 3. Ollama
echo.
echo [3/4] Verificando Ollama...
call :find_ollama
if defined OLLAMA goto ollama_ok

echo Ollama nao encontrado. Baixando o instalador oficial (ollama.com)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://ollama.com/install.ps1 | iex"
call :find_ollama
if defined OLLAMA goto ollama_ok
echo.
echo ERRO: nao foi possivel instalar o Ollama automaticamente.
echo Instale em https://ollama.com e rode este instalador novamente.
pause
exit /b 1

:ollama_ok
call :start_ollama
echo Baixando o modelo llama3.2 (se ainda nao existir)...
%OLLAMA% pull llama3.2
if errorlevel 1 (
    echo ERRO ao baixar o modelo llama3.2. Verifique a internet e rode novamente.
    pause
    exit /b 1
)

rem -------------------------------------------------------------- 4. indexacao
echo.
echo [4/4] Indexando a pasta knowledge (baixa o modelo de embeddings na primeira vez)...
".venv\Scripts\python.exe" scripts\indexar.py
if errorlevel 1 (
    echo ERRO na indexacao. Verifique a internet e rode novamente.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo   INSTALACAO CONCLUIDA - agora execute 02_INICIAR.bat
echo ==========================================================
pause
exit /b 0


rem ================================================================ funcoes
:find_python
rem Python suportado (3.10 a 3.12): lancador "py", "python" do PATH ou instalacao recente do winget.
set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
exit /b 0

:find_ollama
rem O PATH desta janela nao enxerga um Ollama recem-instalado; por isso tambem procura no local padrao.
set "OLLAMA="
where ollama >nul 2>&1 && set "OLLAMA=ollama"
if not defined OLLAMA if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set OLLAMA="%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
exit /b 0

:start_ollama
powershell -NoProfile -Command "try { Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 exit /b 0
echo Iniciando o Ollama em segundo plano...
start "" /min %OLLAMA% serve
timeout /t 5 /nobreak >nul
exit /b 0
