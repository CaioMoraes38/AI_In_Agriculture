import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from contextlib import asynccontextmanager
from services import load_model
from routeIrrigationApi import router as irrigation_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 Iniciando Agricultura IA - Irrigation Prediction API...")
    load_model()
    
    yield
    
    print("\n🔌 API encerrada")


app = FastAPI(
    title="Agricultura IA - Irrigation Prediction API",
    description="API para predição de necessidade de irrigação baseada em Machine Learning",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(irrigation_router)


@app.get("/")
async def root():

    return {
        "message": "Bem-vindo à Agricultura IA",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "irrigation": "/irrigation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    