from pydantic import BaseModel, Field
from .schemas import (
    SoilTypeEnum,
    CropTypeEnum,
    CropGrowthStageEnum,
    SeasonEnum,
    IrrigationTypeEnum,
    WaterSourceEnum,
    MulchingEnum,
)


class PredictionRequest(BaseModel):
    Soil_Type: SoilTypeEnum = Field(..., description="Tipo de solo")
    Soil_pH: float = Field(..., ge=0, le=14, description="pH do solo (0-14)")
    Soil_Moisture: float = Field(..., ge=0, description="Umidade do solo (%)")
    Organic_Carbon: float = Field(..., ge=0, description="Carbono orgânico")
    Electrical_Conductivity: float = Field(..., ge=0, description="Condutividade elétrica")
    Temperature_C: float = Field(..., ge=-50, le=60, description="Temperatura em °C")
    Humidity: float = Field(..., ge=0, le=100, description="Umidade relativa (%)")
    Rainfall_mm: float = Field(..., ge=0, description="Precipitação (mm)")
    Sunlight_Hours: float = Field(..., ge=0, le=24, description="Horas de luz solar")
    Wind_Speed_kmh: float = Field(..., ge=0, description="Velocidade do vento (km/h)")
    Crop_Type: CropTypeEnum = Field(..., description="Tipo de cultura")
    Crop_Growth_Stage: CropGrowthStageEnum = Field(..., description="Estágio de crescimento")
    Season: SeasonEnum = Field(..., description="Estação do ano")
    Irrigation_Type: IrrigationTypeEnum = Field(..., description="Tipo de irrigação")
    Water_Source: WaterSourceEnum = Field(..., description="Fonte de água")
    Field_Area_hectare: float = Field(..., gt=0, description="Área do campo (hectares)")
    Mulching_Used: MulchingEnum = Field(..., description="Uso de cobertura morta")
    Previous_Irrigation_mm: float = Field(..., ge=0, description="Irrigação anterior (mm)")
    Region: int = Field(..., ge=0, description="ID da região")


class PredictionResponse(BaseModel):
    status: str = Field(..., description="Status da requisição")
    prediction_mm: float = Field(..., description="Necessidade de irrigação em mm/dia")
    confidence: str = Field(..., description="Confiança da predição (Alta/Média/Baixa)")
    message: str = Field(..., description="Mensagem descritiva")
