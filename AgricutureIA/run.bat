@echo off
REM Script de inicialização rápida - PlantVision (Windows)

echo 🌱 PlantVision - Inicialização Rápida
echo =======================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python não encontrado
    exit /b 1
)

echo ✅ Python encontrado: 
python --version

REM Navegar para src
cd src

REM Verificar virtualenv
if not exist "..\PlantVision\Scripts\activate.bat" (
    echo.
    echo ⚠️  Virtual environment não encontrado
    echo Criando virtual environment...
    python -m venv ..\PlantVision
)

REM Ativar virtualenv
call ..\PlantVision\Scripts\activate.bat

echo ✅ Virtual environment ativado

REM Instalar dependências
echo.
echo 📦 Instalando dependências...
pip install -r requirements.txt >nul 2>&1

echo.
echo 🚀 Iniciando API...
echo.
echo 📖 Documentação: http://localhost:8000/docs
echo 🔗 API disponível em: http://localhost:8000
echo.
echo Pressione Ctrl+C para parar
echo.

REM Iniciar aplicação
python main.py
