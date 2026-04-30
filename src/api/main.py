import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from contextlib import asynccontextmanager
from services import load_model
from routeIrrigationApi import router as irrigation_router
from routeVisionApi import router as vision_router, load_vision_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 Iniciando Agricultura IA - API Completa...")
    print("   📊 Carregando modelo de irrigação...")
    load_model()
    print("   🧠 Carregando modelo de visão...")
    load_vision_model()
    print("   ✓ Todos os modelos carregados com sucesso!\n")
    
    yield
    
    print("\n🔌 API encerrada")


app = FastAPI(
    title="Agricultura IA - Sistema Completo",
    description="API para predição de irrigação e detecção de doenças em plantas",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(irrigation_router)
app.include_router(vision_router)


@app.get("/")
async def root():
    return {
        "message": "Bem-vindo à Agricultura IA",
        "version": "1.0.0",
        "description": "Sistema completo de IA para agricultura inteligente",
        "endpoints": {
            "docs": "/docs",
            "irrigation": "/irrigation",
            "vision": "/vision"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    