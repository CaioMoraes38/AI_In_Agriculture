"""
Serviço de cálculo de irrigação baseado em condições climáticas
"""
from typing import Dict
from config import (
    AGUA_BASE_LITROS,
    TEMP_LIMITE_SUPERIOR,
    AGUA_EXTRA_TEMP_ALTA,
    UMIDADE_LIMITE,
    AGUA_REDUCAO_UMIDADE,
    CHUVA_LIMITE,
    AGUA_REDUCAO_CHUVA
)


class IrrigationService:
    """Serviço de cálculo inteligente de irrigação"""
    
    def calcular_irrigacao(
        self,
        temperatura: float,
        umidade_solo: float,
        probabilidade_chuva: float
    ) -> Dict:
        """
        Calcula volume de água recomendado baseado em condições climáticas
        
        Args:
            temperatura: Temperatura em Celsius
            umidade_solo: Umidade do solo em percentual (0-100)
            probabilidade_chuva: Probabilidade de chuva em percentual (0-100)
            
        Returns:
            Dicionário com volume recomendado e análise detalhada
        """
        agua_recomendada = AGUA_BASE_LITROS
        fatores_aplicados = []
        
        # Fator de temperatura
        if temperatura > TEMP_LIMITE_SUPERIOR:
            agua_recomendada += AGUA_EXTRA_TEMP_ALTA
            fatores_aplicados.append({
                "fator": "Temperatura alta",
                "valor": temperatura,
                "ajuste": f"+{AGUA_EXTRA_TEMP_ALTA}L"
            })
        
        # Fator de umidade do solo
        if umidade_solo > UMIDADE_LIMITE:
            agua_recomendada -= AGUA_REDUCAO_UMIDADE
            fatores_aplicados.append({
                "fator": "Solo úmido",
                "valor": umidade_solo,
                "ajuste": f"-{AGUA_REDUCAO_UMIDADE}L"
            })
        
        # Fator de probabilidade de chuva
        if probabilidade_chuva > CHUVA_LIMITE:
            agua_recomendada -= AGUA_REDUCAO_CHUVA
            fatores_aplicados.append({
                "fator": "Chance de chuva",
                "valor": probabilidade_chuva,
                "ajuste": f"-{AGUA_REDUCAO_CHUVA}L"
            })
        
        # Garantir que o volume não seja negativo
        agua_final = max(0.0, agua_recomendada)
        
        # Determinar nível de alerta
        nivel_alerta = self._determinar_alerta(temperatura, umidade_solo, probabilidade_chuva)
        
        return {
            "volume_litros": round(agua_final, 2),
            "volume_base_litros": AGUA_BASE_LITROS,
            "fatores_aplicados": fatores_aplicados,
            "nivel_alerta": nivel_alerta,
            "condicoes_entrada": {
                "temperatura_celsius": temperatura,
                "umidade_solo_percentual": umidade_solo,
                "probabilidade_chuva_percentual": probabilidade_chuva
            },
            "status": "sucesso"
        }
    
    @staticmethod
    def _determinar_alerta(
        temperatura: float,
        umidade_solo: float,
        probabilidade_chuva: float
    ) -> str:
        """
        Determina o nível de alerta baseado nas condições
        
        Returns:
            Nível de alerta: 'crítico', 'aviso', 'normal', 'chuva_prevista'
        """
        if probabilidade_chuva > CHUVA_LIMITE:
            return "chuva_prevista"
        
        if temperatura > 35 and umidade_solo < 30:
            return "crítico"
        
        if temperatura > TEMP_LIMITE_SUPERIOR or umidade_solo < 40:
            return "aviso"
        
        return "normal"
