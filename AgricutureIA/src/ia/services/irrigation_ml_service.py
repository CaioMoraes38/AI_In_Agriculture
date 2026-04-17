"""
Serviço de irrigação baseado em Machine Learning
Oferece previsões de necessidade de irrigação usando Random Forest
Suporta predições com 3 variáveis principais ou com dados completos
"""
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


class IrrigationMLService:
    """Serviço de previsão de irrigação baseado em ML"""
    
    # Mapeamento de necessidade de irrigação para volume
    IRRIGACAO_POR_CLASSE = {
        'Low': 5.0,      # 5 litros (pouca necessidade)
        'Medium': 15.0,  # 15 litros (necessidade moderada)
        'High': 25.0     # 25 litros (alta necessidade)
    }
    
    # Variáveis principais que sempre devem estar presentes
    VARIAVEIS_PRINCIPAIS = ['Temperature_C', 'Humidity', 'Rainfall_mm']
    
    # Variáveis numéricas adicionais
    VARIAVEIS_NUMERICAS = [
        'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Sunlight_Hours', 'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
    ]
    
    # Variáveis categóricas
    VARIAVEIS_CATEGORICAS = [
        'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 
        'Irrigation_Type', 'Water_Source', 'Region', 'Mulching_Used'
    ]
    
    def __init__(self, modelos_dir: Path = None):
        """
        Inicializa o serviço de ML para irrigação
        
        Args:
            modelos_dir: Diretório onde estão os arquivos do modelo
        """
        if modelos_dir is None:
            modelos_dir = Path(__file__).parent.parent / 'models'
        
        self.modelos_dir = modelos_dir
        self.modelo = None
        self.scaler = None
        self.encoders = {}
        self.features_info = {}
        self.model_info = {}
        self.modelo_carregado = False
        
        self._carregar_modelo()
    
    def _carregar_modelo(self):
        """Carrega o modelo e artefatos treinados"""
        try:
            # Carregar modelo
            modelo_path = self.modelos_dir / 'irrigation_model.pkl'
            if modelo_path.exists():
                with open(modelo_path, 'rb') as f:
                    self.modelo = pickle.load(f)
            
            # Carregar scaler
            scaler_path = self.modelos_dir / 'irrigation_scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Carregar encoders
            encoders_path = self.modelos_dir / 'irrigation_encoders.json'
            if encoders_path.exists():
                with open(encoders_path, 'r') as f:
                    self.encoders = json.load(f)
            
            # Carregar informações de features
            features_path = self.modelos_dir / 'irrigation_features.json'
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.features_info = json.load(f)
            
            # Carregar informações do modelo
            info_path = self.modelos_dir / 'irrigation_model_info.json'
            if info_path.exists():
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
            
            if self.modelo and self.scaler and self.encoders:
                self.modelo_carregado = True
                print("✓ Modelo de irrigação ML carregado com sucesso")
            else:
                print("⚠️ Modelo de irrigação ML não completamente carregado")
        
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo de irrigação: {e}")
            self.modelo_carregado = False
    
    def prever_irrigacao(
        self,
        temperatura: float,
        umidade: float,
        chuva: float,
        **kwargs
    ) -> Dict:
        """
        Prediz a necessidade de irrigação baseado em dados
        
        Suporta predição com apenas 3 variáveis (temperatura, umidade, chuva)
        ou com variáveis adicionais para maior precisão
        
        Args:
            temperatura: Temperatura em Celsius
            umidade: Umidade relativa em %
            chuva: Precipitação em mm
            **kwargs: Variáveis adicionais opcionais (soil_type, crop_type, etc.)
        
        Returns:
            Dicionário com previsão, confiança e recomendação de volume
        """
        if not self.modelo_carregado:
            # Fallback para cálculo simples se modelo não está disponível
            return self._prever_simples(temperatura, umidade, chuva)
        
        try:
            # Preparar dados para predição
            features = self._preparar_features_predicao(temperatura, umidade, chuva, kwargs)
            
            if features is None:
                return self._prever_simples(temperatura, umidade, chuva)
            
            # Normalizar
            features_scaled = self.scaler.transform([features])
            
            # Prever
            predicao_idx = self.modelo.predict(features_scaled)[0]
            probabilidades = self.modelo.predict_proba(features_scaled)[0]
            
            # Mapear para classe
            classes = self.model_info.get('classes', ['Low', 'Medium', 'High'])
            classe_prevista = classes[predicao_idx]
            confianca = float(probabilidades[predicao_idx])
            
            # Calcular volume recomendado (apenas ML, sem ajustes manuais)
            volume_litros = self.IRRIGACAO_POR_CLASSE.get(classe_prevista, 15.0)
            
            # Determinar nível de alerta
            nivel_alerta = self._determinar_alerta(temperatura, umidade, chuva, classe_prevista)
            
            # Variáveis utilizadas
            variaveis_utilizadas = ['temperatura', 'umidade', 'chuva']
            variaveis_utilizadas.extend([k for k in kwargs.keys()])
            
            return {
                'classe_prevista': classe_prevista,
                'confianca': round(confianca * 100, 2),
                'volume_litros': round(volume_litros, 2),
                'nivel_alerta': nivel_alerta,
                'variaveis_utilizadas': variaveis_utilizadas,
                'condicoes_entrada': {
                    'temperatura_c': temperatura,
                    'umidade_percentual': umidade,
                    'chuva_mm': chuva,
                    'variaveis_adicionais': kwargs
                },
                'modelo_usado': 'random_forest_ml_puro',
                'status': 'sucesso'
            }
        
        except Exception as e:
            print(f"⚠️ Erro na predição ML: {e}")
            return self._prever_simples(temperatura, umidade, chuva)
    
    def _preparar_features_predicao(
        self,
        temperatura: float,
        umidade: float,
        chuva: float,
        dados_adicionais: Dict
    ) -> Optional[List[float]]:
        """
        Prepara features para predição
        
        Args:
            temperatura: Temperatura
            umidade: Umidade
            chuva: Chuva
            dados_adicionais: Dict com dados adicionais
        
        Returns:
            Lista de features na ordem correta ou None se erro
        """
        try:
            features_dict = {}
            
            # Adicionar variáveis principais
            features_dict['Temperature_C'] = temperatura
            features_dict['Humidity'] = umidade
            features_dict['Rainfall_mm'] = chuva
            
            # Adicionar variáveis numéricas adicionais
            for var in self.VARIAVEIS_NUMERICAS:
                if var in dados_adicionais:
                    features_dict[var] = dados_adicionais[var]
                else:
                    # Usar valor médio do dataset
                    features_dict[var] = 0.0
            
            # Processar variáveis categóricas
            for var in self.VARIAVEIS_CATEGORICAS:
                if var in dados_adicionais:
                    valor = str(dados_adicionais[var])
                    # Codificar
                    if var in self.encoders:
                        classes = self.encoders[var]['classes']
                        if valor in classes:
                            features_dict[var] = classes.index(valor)
                        else:
                            features_dict[var] = 0  # Usar primeira classe como padrão
                    else:
                        features_dict[var] = 0
                else:
                    features_dict[var] = 0
            
            # Ordenar na ordem das features do modelo
            features_ordem = self.features_info.get('todas_as_features', 
                                                    list(features_dict.keys()))
            
            features_list = []
            for feat in features_ordem:
                if feat in features_dict:
                    features_list.append(features_dict[feat])
                else:
                    features_list.append(0.0)
            
            return features_list
        
        except Exception as e:
            print(f"⚠️ Erro ao preparar features: {e}")
            return None
    
    def _determinar_alerta(
        self,
        temperatura: float,
        umidade: float,
        chuva: float,
        classe_prevista: str
    ) -> str:
        """
        Determina nível de alerta
        
        Returns:
            Nível de alerta: 'crítico', 'aviso', 'normal', 'chuva_prevista'
        """
        if chuva > 50:
            return 'chuva_prevista'
        
        if classe_prevista == 'High' and temperatura > 35:
            return 'crítico'
        
        if classe_prevista == 'High' or (temperatura > 32 and umidade < 40):
            return 'aviso'
        
        return 'normal'
    
    def _prever_simples(
        self,
        temperatura: float,
        umidade: float,
        chuva: float
    ) -> Dict:
        """
        Previsão simples baseada em regras (fallback)
        Usado quando o modelo ML não está disponível
        """
        # Determinar classe baseada em regras
        if chuva > 100:
            classe = 'Low'
        elif umidade > 75:
            classe = 'Low'
        elif temperatura > 35 and umidade < 40:
            classe = 'High'
        elif temperatura > 30 or umidade < 50:
            classe = 'Medium'
        else:
            classe = 'Low'
        
        volume_base = self.IRRIGACAO_POR_CLASSE.get(classe, 15.0)
        
        # Ajustes adicionais
        ajustes = self._calcular_ajustes(temperatura, umidade, chuva)
        volume_final = max(0.0, volume_base + ajustes['ajuste_total'])
        
        return {
            'classe_prevista': classe,
            'confianca': 75.0,
            'volume_litros': round(volume_final, 2),
            'volume_base': volume_base,
            'ajustes_aplicados': ajustes['detalhes'],
            'nivel_alerta': self._determinar_alerta(temperatura, umidade, chuva, classe),
            'variaveis_utilizadas': ['temperatura', 'umidade', 'chuva'],
            'condicoes_entrada': {
                'temperatura_c': temperatura,
                'umidade_percentual': umidade,
                'chuva_mm': chuva,
                'variaveis_adicionais': {}
            },
            'modelo_usado': 'regras_simples',
            'status': 'sucesso'
        }
    
    def obter_info_modelo(self) -> Dict:
        """Retorna informações sobre o modelo carregado"""
        return {
            'modelo_carregado': self.modelo_carregado,
            'tipo_modelo': self.model_info.get('tipo', 'Não disponível'),
            'acurácia': self.model_info.get('accuracy', 'Não disponível'),
            'classes': self.model_info.get('classes', []),
            'número_features': self.features_info.get('total_features', 0),
            'features_principais': self.VARIAVEIS_PRINCIPAIS,
            'features_adicionais_disponíveis': self.VARIAVEIS_NUMERICAS + self.VARIAVEIS_CATEGORICAS
        }
    
    def listar_parametros_opcionais(self) -> Dict:
        """Retorna lista de parâmetros opcionais para melhor predição"""
        return {
            'variaveis_numéricas': self.VARIAVEIS_NUMERICAS,
            'variaveis_categóricas': self.VARIAVEIS_CATEGORICAS,
            'descrição': 'Envie essas variáveis adicionais para melhorar a precisão da predição',
            'exemplo_completo': {
                'temperatura': 28.5,
                'umidade': 65.0,
                'chuva': 15.0,
                'soil_type': 'Clay',
                'crop_type': 'Wheat',
                'crop_growth_stage': 'Vegetative',
                'season': 'Rabi',
                'soil_moisture': 45.0,
                'sunlight_hours': 8.0
            }
        }
