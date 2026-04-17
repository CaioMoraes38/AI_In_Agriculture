from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from api.dependencies import Dependencias
from ia.services.irrigation_ml_regression_service import IrrigationMLRegressionService

router = APIRouter(prefix="/irrigacao", tags=["Irrigação ML"])


def obter_irrigation_ml_service() -> IrrigationMLRegressionService:
    return Dependencias.obter_irrigation_ml_service()


@router.post("/ml/prever", summary="Prever necessidade de irrigação com ML")
async def prever_irrigacao_ml(
    temperatura: float,
    umidade: float,
    chuva: float,
    soil_type: Optional[str] = None,
    crop_type: Optional[str] = None,
    crop_growth_stage: Optional[str] = None,
    season: Optional[str] = None,
    soil_moisture: Optional[float] = None,
    soil_ph: Optional[float] = None,
    sunlight_hours: Optional[float] = None,
    wind_speed: Optional[float] = None,
    service: IrrigationMLRegressionService = Depends(obter_irrigation_ml_service)
):
    try:
        kwargs = {}
        if soil_type:
            kwargs['Soil_Type'] = soil_type
        if crop_type:
            kwargs['Crop_Type'] = crop_type
        if crop_growth_stage:
            kwargs['Crop_Growth_Stage'] = crop_growth_stage
        if season:
            kwargs['Season'] = season
        if soil_moisture is not None:
            kwargs['Soil_Moisture'] = soil_moisture
        if soil_ph is not None:
            kwargs['Soil_pH'] = soil_ph
        if sunlight_hours is not None:
            kwargs['Sunlight_Hours'] = sunlight_hours
        if wind_speed is not None:
            kwargs['Wind_Speed_kmh'] = wind_speed
        
        resultado = service.prever_irrigacao(
            temperatura=temperatura,
            umidade=umidade,
            chuva=chuva,
            **kwargs
        )
        return resultado
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erro na previsão de irrigação: {str(e)}"
        )


@router.get("/ml/info", summary="Informações do modelo ML")
async def info_modelo_ml(
    service: IrrigationMLRegressionService = Depends(obter_irrigation_ml_service)
):
    return service.obter_info_modelo()


@router.get("/ml/parametros", summary="Parâmetros opcionais disponíveis")
async def parametros_ml(
    service: IrrigationMLRegressionService = Depends(obter_irrigation_ml_service)
):
    return service.listar_parametros_opcionais()


@router.post("/ml/exemplo", summary="Exemplo de previsão com ML")
async def exemplo_ml(
    service: IrrigationMLRegressionService = Depends(obter_irrigation_ml_service)
):
    return service.prever_irrigacao(
        temperatura=28.5,
        umidade=65.0,
        chuva=15.0,
        soil_type='Clay',
        crop_type='Wheat',
        crop_growth_stage='Vegetative'
    )

