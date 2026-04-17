import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import base64
from io import BytesIO
import warnings
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

try:
    from PIL import Image
    from tensorflow import keras
    VISAO_DISPONIVEL = True
except ImportError:
    VISAO_DISPONIVEL = False
    keras = None
    Image = None


class PlantDiseaseService:
    IMG_SIZE = (224, 224)
    
    def __init__(self, modelos_dir: Path = None):
        if modelos_dir is None:
            modelos_dir = Path(__file__).parent.parent / 'models'
        
        self.modelos_dir = modelos_dir
        self.modelo = None
        self.classes = []
        self.modelo_carregado = False
        
        self._carregar_modelo()
    
    def _carregar_modelo(self):
        if not VISAO_DISPONIVEL:
            print("[AVISO] TensorFlow/Keras nao disponivel - servico de visao desativado")
            return
        
        try:
            classes_path = self.modelos_dir / 'classes_plantas.json'
            if classes_path.exists():
                with open(classes_path, 'r') as f:
                    classes_raw = json.load(f)
                    self.classes = [c.replace('Tomato___', '') for c in classes_raw]
            
            modelo_path = self.modelos_dir / 'model_Tomato.keras'
            if modelo_path.exists():
                self.modelo = keras.models.load_model(modelo_path)
                self.modelo_carregado = True
                
            else:
                print(f"[AVISO] Modelo nao encontrado: {modelo_path}")
        
        except Exception as e:
            print(f"[ERRO] Erro ao carregar modelo de visao: {e}")
            self.modelo_carregado = False
    
    def _processar_imagem(self, imagem_bytes: bytes) -> Optional[np.ndarray]:
        try:
            img = Image.open(BytesIO(imagem_bytes))
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize(self.IMG_SIZE)
            img_array = np.array(img, dtype=np.float32)
            
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            green_mask = (g > r * 1.1) & (g > b * 1.1)
            
            percentual_verde = np.sum(green_mask) / (224 * 224)
            
            if percentual_verde < 0.15:
                img_array = gaussian_filter(img_array, sigma=1.0)
            
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        
        except Exception as e:
            print(f"[ERRO] Erro ao processar imagem: {e}")
            return None
    
    def detectar_doenca(self, imagem_base64: str) -> Dict:
        if not self.modelo_carregado:
            return {
                'status': 'erro',
                'mensagem': 'Modelo de visão não disponível',
                'sugestao': 'Certifique-se de que TensorFlow está instalado'
            }
        
        try:
            try:
                imagem_bytes = base64.b64decode(imagem_base64)
            except Exception:
                return {
                    'status': 'erro',
                    'mensagem': 'Imagem base64 inválida'
                }
            
            img_array = self._processar_imagem(imagem_bytes)
            if img_array is None:
                return {
                    'status': 'erro',
                    'mensagem': 'Falha ao processar imagem'
                }
            
            predicoes = self.modelo.predict(img_array, verbose=0)
            confiancas = predicoes[0]
            
            idx_max = np.argmax(confiancas)
            classe_prevista = self.classes[idx_max]
            confianca = float(confiancas[idx_max])
            
            indices_top3 = np.argsort(confiancas)[::-1][:3]
            top3 = [
                {
                    'classe': self.classes[i],
                    'confianca': round(float(confiancas[i]) * 100, 2)
                }
                for i in indices_top3
            ]
            
            eh_saudavel = 'healthy' in classe_prevista.lower()
            recomendacoes = self._gerar_recomendacoes(classe_prevista, eh_saudavel)
            
            return {
                'classe_prevista': classe_prevista,
                'eh_saudavel': eh_saudavel,
                'confianca_percentual': round(confianca * 100, 2),
                'top_3_predicoes': top3,
                'recomendacoes': recomendacoes,
                'modelo_usado': 'cnn_keras_tomato',
                'tamanho_imagem': f'{self.IMG_SIZE[0]}x{self.IMG_SIZE[1]}',
                'status': 'sucesso'
            }
        
        except Exception as e:
            print(f"[ERRO] Erro na deteccao: {e}")
            return {
                'status': 'erro',
                'mensagem': f'Erro ao processar: {str(e)}'
            }
    
    def _gerar_recomendacoes(self, classe: str, eh_saudavel: bool) -> Dict:
        if eh_saudavel:
            return {
                'status': 'planta_saudavel',
                'mensagem': 'Planta em bom estado de saúde',
                'acoes': [
                    'Manter rotina de irrigação regular',
                    'Continuar com manejo adequado',
                    'Monitorar periodicamente'
                ]
            }
        
        recomendacoes_doencas = {
            'Bacterial_spot': {
                'causa': 'Infecção bacteriana',
                'severidade': 'Alta',
                'acoes': [
                    'Aplicar fungicida/bactericida',
                    'Aumentar espaçamento entre plantas',
                    'Reduzir umidade foliar',
                    'Remover folhas infectadas'
                ]
            },
            'Early_blight': {
                'causa': 'Infecção fúngica (Alternaria)',
                'severidade': 'Média',
                'acoes': [
                    'Remover folhas basais infectadas',
                    'Aplicar fungicida preventivo',
                    'Melhorar circulação de ar',
                    'Evitar molhar a folhagem'
                ]
            },
            'Late_blight': {
                'causa': 'Infecção fúngica (Phytophthora)',
                'severidade': 'Muito Alta',
                'acoes': [
                    'Aplicar fungicida imediatamente',
                    'Remover folhas e frutos infectados',
                    'Destruir plantas severamente afetadas',
                    'Melhorar drenagem do solo'
                ]
            },
            'Leaf_Mold': {
                'causa': 'Infecção fúngica (Cladosporium)',
                'severidade': 'Média',
                'acoes': [
                    'Reduzir umidade (aumentar ventilação)',
                    'Aplicar fungicida',
                    'Remover folhas infectadas',
                    'Evitar molhar a folhagem'
                ]
            },
            'Septoria_leaf_spot': {
                'causa': 'Infecção fúngica (Septoria)',
                'severidade': 'Média',
                'acoes': [
                    'Remover folhas infectadas',
                    'Aplicar fungicida de cobre',
                    'Melhorar circulação de ar',
                    'Desinfetar ferramentas'
                ]
            },
            'Spider_mites': {
                'causa': 'Infestação de ácaros',
                'severidade': 'Média',
                'acoes': [
                    'Aumentar umidade (borrifar água)',
                    'Aplicar acaricida/inseticida',
                    'Remover folhas severamente afetadas',
                    'Monitorar plantas vizinhas'
                ]
            },
            'Target_Spot': {
                'causa': 'Infecção fúngica (Corynespora)',
                'severidade': 'Alta',
                'acoes': [
                    'Remover folhas infectadas',
                    'Aplicar fungicida de cobre',
                    'Melhorar circulação de ar',
                    'Reduzir umidade foliar'
                ]
            },
            'Tomato_Yellow_Leaf_Curl_Virus': {
                'causa': 'Infecção viral (TYLCV)',
                'severidade': 'Muito Alta',
                'acoes': [
                    'Remover e destruir planta infectada',
                    'Controlar vetores (moscas-brancas)',
                    'Aplicar inseticida',
                    'Isolar plantas vizinhas'
                ]
            },
            'Tomato_mosaic_virus': {
                'causa': 'Infecção viral (TMV)',
                'severidade': 'Alta',
                'acoes': [
                    'Remover e destruir planta infectada',
                    'Desinfetar ferramentas com álcool',
                    'Evitar fumar perto das plantas',
                    'Usar sementes certificadas'
                ]
            },
            'healthy': {
                'causa': 'Nenhuma doença detectada',
                'severidade': 'Nenhuma',
                'acoes': [
                    'Manter rotina de irrigação regular',
                    'Continuar com manejo adequado',
                    'Monitorar periodicamente'
                ]
            }
        }
        
        for chave, rec in recomendacoes_doencas.items():
            if chave in classe:
                return {
                    'status': 'doenca_detectada',
                    'doenca': classe,
                    **rec
                }
        
        return {
            'status': 'doenca_desconhecida',
            'mensagem': f'Doença {classe} não reconhecida',
            'acoes': [
                'Consultar especialista em fitopatologia',
                'Enviar amostra para análise laboratorial'
            ]
        }
    
    def obter_info(self) -> Dict:
        return {
            'tipo_modelo': 'CNN (Convolutional Neural Network)',
            'nome': 'Modelo de Detecção de Doenças em Tomate',
            'descricao': 'Classifica doenças em imagens de folhas de tomate',
            'classes_disponiveis': self.classes,
            'tamanho_imagem': f'{self.IMG_SIZE[0]}x{self.IMG_SIZE[1]}',
            'n_classes': len(self.classes),
            'status': 'operacional' if self.modelo_carregado else 'inoperável'
        }
