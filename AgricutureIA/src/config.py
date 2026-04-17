import os
from pathlib import Path

DIRETORIO_SRC = Path(__file__).parent
DIRETORIO_RAIZ = DIRETORIO_SRC.parent
DIRETORIO_MODELOS = DIRETORIO_SRC / 'models'

API_TITULO = "PlantVision AI - TCC"
API_DESCRICAO = "API de Inteligência Artificial para Previsão de Irrigação"
API_VERSAO = "2.0.0"
API_CORS_ORIGINS = ["*"]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
