@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Criando ambiente virtual (.venv)...
python -m venv .venv
if errorlevel 1 (
  echo.
  echo Nao foi possivel criar o ambiente. Verifique se o Python 3.10+ esta instalado e no PATH.
  pause
  exit /b 1
)
echo Instalando bibliotecas (NumPy, OpenCV, scikit-learn)...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Falha na instalacao das bibliotecas. Verifique a conexao com a internet.
  pause
  exit /b 1
)
echo.
echo Instalacao concluida. Use executar_interface.bat para abrir o ContAedes.
pause
