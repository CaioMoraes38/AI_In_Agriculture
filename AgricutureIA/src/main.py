from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from config import API_TITULO, API_DESCRICAO, API_VERSAO, API_CORS_ORIGINS
from api.dependencies import Dependencias
from api.routes import irrigation
from api.routes import diseases


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title=API_TITULO,
    description=API_DESCRICAO,
    version=API_VERSAO,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(irrigation.router)
app.include_router(diseases.router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "nome": API_TITULO,
        "versao": API_VERSAO,
        "descricao": API_DESCRICAO,
        "endpoints": {
            "visao": "/docs#/Doencas de Plantas",
            "irrigacao": "/docs#/Irrigacao",
            "documentacao": "/docs",
            "alternativa": "/redoc"
        }
    }


@app.get("/saude", tags=["Health"])
async def verificar_saude():
    return {
        "status": "ativo",
        "api": "PlantVision",
        "versao": API_VERSAO,
        "servicos": {
            "irrigacao": "pronto",
            "doencas": "pronto"
        }
    }


def customizar_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=API_TITULO,
        version=API_VERSAO,
        description=API_DESCRICAO,
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = customizar_openapi


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
