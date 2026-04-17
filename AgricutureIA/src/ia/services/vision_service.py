"""
Serviço de análise de visão computacional para plantas
"""
import numpy as np
from typing import Dict, Optional
from ...ia.preprocessing.image_processor import ImageProcessor
from ...ia.models.model_loader import ModeloVisaoLoader, ClassesLoader


class VisionService:
    """Serviço de análise visual de plantas usando IA"""
    
    def __init__(self, modelo_loader: ModeloVisaoLoader, classes_loader: ClassesLoader):
        """
        Inicializa o serviço de visão
        
        Args:
            modelo_loader: Loader do modelo treinado
            classes_loader: Loader das classes
        """
        self.modelo_loader = modelo_loader
        self.classes_loader = classes_loader
        self.image_processor = ImageProcessor()
    
    def analisar(self, conteudo_imagem: bytes) -> Dict:
        """
        Analisa uma imagem e retorna diagnóstico da planta
        
        Args:
            conteudo_imagem: Bytes da imagem
            
        Returns:
            Dicionário com diagnóstico e confiança
            
        Raises:
            ValueError: Se o modelo não estiver carregado
            Exception: Se houver erro no processamento
        """
        if self.modelo_loader.modelo is None:
            raise ValueError("Modelo de visão não foi carregado")
        
        if self.classes_loader.nomes_classes is None:
            raise ValueError("Classes não foram carregadas")
        
        try:
            # Processar imagem
            imagem_processada = self.image_processor.processar_do_bytes(conteudo_imagem)
            
            # Fazer predição
            predicoes = self.modelo_loader.modelo.predict(imagem_processada, verbose=0)
            
            # Extrair resultados
            indice_melhor = np.argmax(predicoes[0])
            confianca = float(np.max(predicoes[0])) * 100
            diagnostico = self.classes_loader.nomes_classes[indice_melhor]
            
            # Pegar top 3 previsões
            top_3_indices = np.argsort(predicoes[0])[-3:][::-1]
            top_3 = [
                {
                    "diagnostico": self.classes_loader.nomes_classes[idx],
                    "confianca": float(predicoes[0][idx]) * 100
                }
                for idx in top_3_indices
            ]
            
            return {
                "diagnostico_principal": diagnostico,
                "confianca_porcentagem": round(confianca, 2),
                "top_3_previsoes": top_3,
                "status": "sucesso"
            }
            
        except Exception as e:
            raise Exception(f"Erro ao analisar imagem: {str(e)}")
    
    def obter_classes(self) -> list:
        """
        Retorna lista de todas as classes disponíveis
        
        Returns:
            Lista de nomes de classes
        """
        return self.classes_loader.nomes_classes or []
