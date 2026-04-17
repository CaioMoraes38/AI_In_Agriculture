import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


class IrrigationMLRegressionService:
    VARIAVEIS_PRINCIPAIS = ['Temperature_C', 'Humidity', 'Rainfall_mm']
    
    VARIAVEIS_NUMERICAS = [
        'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Sunlight_Hours', 'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
    ]
    
    VARIAVEIS_CATEGORICAS = [
        'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 
        'Irrigation_Type', 'Water_Source', 'Region', 'Mulching_Used'
    ]
    
    def __init__(self, modelos_dir: Path = None):
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
        try:
            modelo_path = self.modelos_dir / 'irrigation_regressor.pkl'
            if modelo_path.exists():
                with open(modelo_path, 'rb') as f:
                    self.modelo = pickle.load(f)
            
            scaler_path = self.modelos_dir / 'irrigation_scaler_regressor.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            encoders_path = self.modelos_dir / 'irrigation_encoders_regressor.json'
            if encoders_path.exists():
                with open(encoders_path, 'r') as f:
                    self.encoders = json.load(f)
            
            features_path = self.modelos_dir / 'irrigation_features_regressor.json'
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.features_info = json.load(f)
            
            info_path = self.modelos_dir / 'irrigation_regressor_info.json'
            if info_path.exists():
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
            
            if self.modelo and self.scaler and self.encoders:
                self.modelo_carregado = True
                print("[OK] Modelo de irrigacao ML (Regressao) carregado com sucesso")
                print(f"  R² Score: {self.model_info.get('r2_score', 'N/A'):.4f}")
                print(f"  MAE: {self.model_info.get('mae', 'N/A'):.3f}L")
            else:
                print("[AVISO] Modelo de irrigacao ML nao completamente carregado")
        
        except Exception as e:
            print(f"[ERRO] Erro ao carregar modelo de irrigacao: {e}")
            self.modelo_carregado = False
    
    def prever_irrigacao(
        self,
        temperatura: float,
        umidade: float,
        chuva: float,
        **kwargs
    ) -> Dict:
        if not self.modelo_carregado:
            return self._prever_padrao(temperatura, umidade, chuva)
        
        try:
            features = self._preparar_features_predicao(temperatura, umidade, chuva, kwargs)
            
            if features is None:
                return self._prever_padrao(temperatura, umidade, chuva)
            
            features_scaled = self.scaler.transform([features])
            
            volume_litros = float(self.modelo.predict(features_scaled)[0])
            volume_litros = max(0.0, volume_litros)
            
            nivel_alerta = self._determinar_alerta_por_volume(volume_litros, umidade, chuva)
            
            variaveis_utilizadas = ['temperatura', 'umidade', 'chuva']
            variaveis_utilizadas.extend([k for k in kwargs.keys()])
            
            return {
                'volume_litros': round(volume_litros, 2),
                'nivel_alerta': nivel_alerta,
                'variaveis_utilizadas': variaveis_utilizadas,
                'condicoes_entrada': {
                    'temperatura_c': temperatura,
                    'umidade_percentual': umidade,
                    'chuva_mm': chuva,
                    'variaveis_adicionais': kwargs
                },
                'modelo_usado': 'random_forest_regressao_ml',
                'r2_score': self.model_info.get('r2_score', None),
                'mae': self.model_info.get('mae', None),
                'status': 'sucesso'
            }
        
        except Exception as e:
            print(f"[ERRO] Erro na predicao ML: {e}")
            return self._prever_padrao(temperatura, umidade, chuva)
    
    def _preparar_features_predicao(
        self,
        temperatura: float,
        umidade: float,
        chuva: float,
        dados_adicionais: Dict
    ) -> Optional[List[float]]:
        try:
            features_dict = {}
            
            features_dict['Temperature_C'] = temperatura
            features_dict['Humidity'] = umidade
            features_dict['Rainfall_mm'] = chuva
            
            for var in self.VARIAVEIS_NUMERICAS:
                if var in dados_adicionais:
                    features_dict[var] = dados_adicionais[var]
                else:
                    features_dict[var] = 0.0
            
            for var in self.VARIAVEIS_CATEGORICAS:
                if var in dados_adicionais:
                    valor = str(dados_adicionais[var])
                    if var in self.encoders:
                        classes = self.encoders[var]['classes']
                        if valor in classes:
                            features_dict[var] = classes.index(valor)
                        else:
                            features_dict[var] = 0
                    else:
                        features_dict[var] = 0
                else:
                    features_dict[var] = 0
            
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
            print(f"[ERRO] Erro ao preparar features: {e}")
            return None
    
    def _determinar_alerta_por_volume(
        self,
        volume: float,
        umidade: float,
        chuva: float
    ) -> str:
        if chuva > 50:
            return 'chuva_prevista'
        
        if umidade > 85:
            return 'normal'
        
        if volume > 25:
            return 'crítico'
        elif volume > 20:
            return 'aviso'
        else:
            return 'normal'
    
    def _prever_padrao(
        self,
        temperatura: float,
        umidade: float,
        chuva: float
    ) -> Dict:
        volume = 15.0
        
        if temperatura > 35:
            volume += 5.0
        elif temperatura > 30:
            volume += 2.5
        elif temperatura < 15:
            volume -= 5.0
        
        if umidade > 80:
            volume -= 7.0
        elif umidade < 40:
            volume += 5.0
        
        if chuva > 50:
            volume -= 8.0
        elif chuva > 10:
            volume -= 3.0
        
        volume = max(0.0, volume)
        
        return {
            'volume_litros': round(volume, 2),
            'nivel_alerta': self._determinar_alerta_por_volume(volume, umidade, chuva),
            'modelo_usado': 'fallback_regras',
            'status': 'aviso_modelo_indisponivel'
        }
    
    def obter_info(self) -> Dict:
        return {
            'tipo_modelo': 'regressão',
            'nome': 'Random Forest Regressor para Irrigação',
            'descricao': 'Prediz volume de água em litros baseado em dados climáticos e do solo',
            'acuracia': 'R² Score: {:.2%}'.format(self.model_info.get('r2_score', 0)),
            'mae': '{:.3f}L'.format(self.model_info.get('mae', 0)),
            'rmse': '{:.3f}L'.format(self.model_info.get('rmse', 0)),
            'features_utilizadas': len(self.features_info.get('todas_as_features', [])),
            'features': self.features_info.get('todas_as_features', []),
            'status': 'operacional' if self.modelo_carregado else 'inoperável'
        }
