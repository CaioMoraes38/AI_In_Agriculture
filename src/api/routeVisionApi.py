from fastapi import APIRouter, File, UploadFile, HTTPException
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from io import BytesIO
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision", tags=["Vision"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/modelVisionV1/modelo_mobilenet_tomate.keras")

IMG_SIZE = (224, 224)
vision_model = None

def load_vision_model():
    global vision_model
    try:
        if os.path.exists(MODEL_PATH):
            vision_model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("✓ Modelo MobileNetV2 carregado com sucesso")
            return True
        logger.error(f"❌ Modelo não encontrado em {MODEL_PATH}")
        return False
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        return False

def preprocess_image(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        image = image.resize(IMG_SIZE)
        
        image_array = np.array(image, dtype=np.float32)
        
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Erro no preprocessamento: {str(e)}")
        raise ValueError("Formato de imagem inválido")

def get_disease_info(index):
    disease_map = {
        0: {"code": "Tomato___Bacterial_spot", "name": "Mancha Bacteriana"},
        1: {"code": "Tomato___Early_blight", "name": "Requeima Precoce"},
        2: {"code": "Tomato___healthy", "name": "Saudável"},
        3: {"code": "Tomato___Late_blight", "name": "Requeima Tardia"},
        4: {"code": "Tomato___Leaf_Mold", "name": "Mofo da Folha"},
        5: {"code": "Tomato___Septoria_leaf_spot", "name": "Mancha Septória"},
        6: {"code": "Tomato___Spider_mites_Two-spotted_spider_mite", "name": "Ácaro Rajado"},
        7: {"code": "Tomato___Target_Spot", "name": "Mancha Alvo"},
        8: {"code": "Tomato___Tomato_mosaic_virus", "name": "Vírus do Mosaico"},
        9: {"code": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "name": "Vírus do Enrolamento"}
    }
    return disease_map.get(index, {"code": "Unknown", "name": "Desconhecido"})

@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if vision_model is None and not load_vision_model():
        raise HTTPException(status_code=503, detail="Modelo não disponível")
    
    try:
        contents = await file.read()
        image_array = preprocess_image(contents)
        
        predictions = vision_model.predict(image_array, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        info = get_disease_info(predicted_class_idx)
        
        return {
            "status": "success",
            "prediction": {
                "id": int(predicted_class_idx),
                "name": info["name"],
                "code": info["code"],
                "confidence": f"{confidence*100:.2f}%"
            },
            "recommendation": "✓ Planta saudável" if "healthy" in info["code"] else f"⚠️ Atenção: {info['name']} detectada."
        }

    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar imagem")

load_vision_model()