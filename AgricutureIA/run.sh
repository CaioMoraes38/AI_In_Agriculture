#!/bin/bash
# Script de inicialização rápida - PlantVision

echo "🌱 PlantVision - Inicialização Rápida"
echo "======================================"
echo ""

# Verificar Python
if ! command -v python &> /dev/null; then
    echo "❌ Python não encontrado"
    exit 1
fi

echo "✅ Python encontrado: $(python --version)"

# Navegar para src
cd src

# Verificar virtualenv
if [ ! -d "../PlantVision" ]; then
    echo ""
    echo "⚠️  Virtual environment não encontrado"
    echo "Criando virtual environment..."
    python -m venv ../PlantVision
fi

# Ativar virtualenv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source ../PlantVision/Scripts/activate
else
    source ../PlantVision/bin/activate
fi

echo "✅ Virtual environment ativado"

# Instalar dependências
echo ""
echo "📦 Instalando dependências..."
pip install -r requirements.txt 2>/dev/null

echo ""
echo "🚀 Iniciando API..."
echo ""
echo "📖 Documentação: http://localhost:8000/docs"
echo "🔗 API disponível em: http://localhost:8000"
echo ""
echo "Pressione Ctrl+C para parar"
echo ""

# Iniciar aplicação
python main.py
