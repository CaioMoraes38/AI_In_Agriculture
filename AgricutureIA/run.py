#!/usr/bin/env python3
"""
Script de inicialização cruzada - PlantVision
Funciona em Windows, macOS e Linux
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🌱 PlantVision - Inicialização")
    print("=" * 50)
    print()
    
    # Verificar Python
    try:
        py_version = sys.version.split()[0]
        print(f"✅ Python {py_version} encontrado")
    except Exception as e:
        print(f"❌ Erro ao verificar Python: {e}")
        return 1
    
    # Navegar para src
    src_dir = Path(__file__).parent / "src"
    os.chdir(src_dir)
    
    print()
    print("🚀 Iniciando API FastAPI...")
    print()
    print("📖 Documentação: http://localhost:8000/docs")
    print("🔗 API disponível em: http://localhost:8000")
    print()
    print("Pressione Ctrl+C para parar")
    print()
    
    try:
        # Importar e executar uvicorn
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except ImportError:
        print("❌ FastAPI/Uvicorn não instalado")
        print("Execute: pip install -r requirements.txt")
        return 1
    except KeyboardInterrupt:
        print()
        print("👋 API encerrada")
        return 0
    except Exception as e:
        print(f"❌ Erro: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
