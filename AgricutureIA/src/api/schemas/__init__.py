"""
Modelos Pydantic para schemas da API
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class DadosIrrigacao(BaseModel):
    """Schema para dados de irrigação"""
    temperatura: float = Field(..., gt=-50, lt=60, description="Temperatura em Celsius")
    umidade_solo: float = Field(..., ge=0, le=100, description="Umidade do solo em %")
    probabilidade_chuva: float = Field(..., ge=0, le=100, description="Probabilidade de chuva em %")
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperatura": 28.5,
                "umidade_solo": 55.0,
                "probabilidade_chuva": 30.0
            }
        }


class FatorAplicado(BaseModel):
    """Schema para fatores aplicados no cálculo"""
    fator: str
    valor: float
    ajuste: str


class CondicoesEntrada(BaseModel):
    """Schema para condições de entrada"""
    temperatura_celsius: float
    umidade_solo_percentual: float
    probabilidade_chuva_percentual: float


class ResultadoIrrigacao(BaseModel):
    """Schema para resultado do cálculo de irrigação"""
    volume_litros: float
    volume_base_litros: float
    fatores_aplicados: List[FatorAplicado]
    nivel_alerta: str = Field(..., description="Nível de alerta: crítico, aviso, normal, chuva_prevista")
    condicoes_entrada: CondicoesEntrada
    status: str


class PrevisaoTopo(BaseModel):
    """Schema para previsão individual"""
    diagnostico: str
    confianca: float


class ResultadoVisao(BaseModel):
    """Schema para resultado da análise de visão"""
    diagnostico_principal: str
    confianca_porcentagem: float
    top_3_previsoes: List[PrevisaoTopo]
    status: str


class ListaClasses(BaseModel):
    """Schema para lista de classes disponíveis"""
    total_classes: int
    classes: List[str]
    status: str


class ErroResponse(BaseModel):
    """Schema para resposta de erro"""
    status: str = "erro"
    detalhe: str
    codigo_erro: Optional[str] = None
