from fastapi import APIRouter, HTTPException
from dtos import PredictionRequest, PredictionResponse
from services import predict_irrigation, get_model_data, get_model_data


router = APIRouter(prefix="/irrigation", tags=["Irrigação"])


@router.get("/")
async def root():

    return {
        "message": "Bem-vindo à API de Predição de Irrigação",
        "model_loaded": get_model_data() is not None,
        "docs": "/docs"
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": get_model_data() is not None
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if get_model_data() is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de ML não carregado. Verifique o arquivo do modelo."
        )
    
    try:
        return predict_irrigation(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Erro de validação: {str(ve)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar predição: {str(e)}")
