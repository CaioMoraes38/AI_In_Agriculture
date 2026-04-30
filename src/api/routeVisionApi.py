from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import pickle
import json
import os
from PIL import Image
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["Vision"])

MODEL_PATH = "src/models/modelVisionV1/vision_model_v1.h5"
METRICS_PATH = "src/models/modelVisionV1/vision_model_metrics.json"
LABELS_PATH = "src/models/modelVisionV1/class_labels.pkl"

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

vision_model = None
class_labels = None
metrics = None

def load_vision_model():
    global vision_model, class_labels, metrics
    
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Modelo não encontrado em {MODEL_PATH}")
            return False
        
        vision_model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("✓ Modelo de visão carregado com sucesso")
        
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'rb') as f:
                class_labels = pickle.load(f)
            logger.info("✓ Labels carregados com sucesso")
        
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            logger.info("✓ Métricas carregadas com sucesso")
        
        return True
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        return False

def preprocess_image(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logger.error(f"Erro ao preprocessar imagem: {str(e)}")
        raise ValueError(f"Erro ao processar imagem: {str(e)}")

def get_disease_name(index):
    disease_map = {
        0: "Tomato___Bacterial_spot",
        1: "Tomato___Early_blight",
        2: "Tomato___healthy",
        3: "Tomato___Late_blight",
        4: "Tomato___Leaf_Mold",
        5: "Tomato___Septoria_leaf_spot",
        6: "Tomato___Spider_mites Two-spotted_spider_mite",
        7: "Tomato___Target_Spot",
        8: "Tomato___Tomato_mosaic_virus",
        9: "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    }
    return disease_map.get(index, "Unknown")

def format_disease_name(disease):
    name_map = {
        "Tomato___Bacterial_spot": "Mancha Bacteriana",
        "Tomato___Early_blight": "Requeima Precoce",
        "Tomato___healthy": "Saudável",
        "Tomato___Late_blight": "Requeima Tardia",
        "Tomato___Leaf_Mold": "Mofo da Folha",
        "Tomato___Septoria_leaf_spot": "Mancha Septória",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Ácaro Rajado",
        "Tomato___Target_Spot": "Mancha Alvo",
        "Tomato___Tomato_mosaic_virus": "Vírus do Mosaico",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Vírus do Enrolamento"
    }
    return name_map.get(disease, disease)

@router.get("/health")
async def health_check():
    if vision_model is None:
        return {
            "status": "error",
            "model_loaded": False,
            "message": "Modelo de visão não carregado"
        }
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "MobileNetV2",
        "classes": 10,
        "image_size": IMG_SIZE
    }

@router.get("/")
async def root():
    if vision_model is None:
        return {
            "status": "error",
            "message": "Modelo não carregado"
        }
    
    return {
        "status": "ready",
        "message": "API de Visão - Detecção de Doenças em Tomate",
        "version": "1.0",
        "endpoints": [
            {"method": "GET", "path": "/vision/health", "description": "Health check"},
            {"method": "GET", "path": "/vision/classes", "description": "Listar classes"},
            {"method": "POST", "path": "/vision/predict", "description": "Predizer doença"},
            {"method": "POST", "path": "/vision/batch", "description": "Predizer múltiplas imagens"}
        ],
        "model_accuracy": metrics.get("test_accuracy") if metrics else "unknown"
    }

@router.get("/classes")
async def get_classes():
    classes = [
        {
            "id": 0,
            "code": "Tomato___Bacterial_spot",
            "name": "Mancha Bacteriana",
            "description": "Infecção bacteriana que causa manchas aquosas nas folhas"
        },
        {
            "id": 1,
            "code": "Tomato___Early_blight",
            "name": "Requeima Precoce",
            "description": "Doença fúngica que afeta principalmente folhas inferiores"
        },
        {
            "id": 2,
            "code": "Tomato___healthy",
            "name": "Saudável",
            "description": "Planta sem sinais visíveis de doença"
        },
        {
            "id": 3,
            "code": "Tomato___Late_blight",
            "name": "Requeima Tardia",
            "description": "Doença devastadora causada por Phytophthora infestans"
        },
        {
            "id": 4,
            "code": "Tomato___Leaf_Mold",
            "name": "Mofo da Folha",
            "description": "Infecção fúngica que causa mofo amarelado nas folhas"
        },
        {
            "id": 5,
            "code": "Tomato___Septoria_leaf_spot",
            "name": "Mancha Septória",
            "description": "Manchas circulares com halo amarelo"
        },
        {
            "id": 6,
            "code": "Tomato___Spider_mites Two-spotted_spider_mite",
            "name": "Ácaro Rajado",
            "description": "Praga que causa descoloração e enrugamento das folhas"
        },
        {
            "id": 7,
            "code": "Tomato___Target_Spot",
            "name": "Mancha Alvo",
            "description": "Manchas concêntricas que lembram um alvo"
        },
        {
            "id": 8,
            "code": "Tomato___Tomato_mosaic_virus",
            "name": "Vírus do Mosaico",
            "description": "Padrão de mosaico nas folhas com descoloração"
        },
        {
            "id": 9,
            "code": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "name": "Vírus do Enrolamento",
            "description": "Folhas ficam amarelas e enroladas"
        }
    ]
    
    return {
        "status": "success",
        "total_classes": len(classes),
        "classes": classes
    }

@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if vision_model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de visão não está carregado"
        )
    
    try:
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Arquivo vazio recebido"
            )
        
        image_array = preprocess_image(contents)
        
        predictions = vision_model.predict(image_array, verbose=0)
        
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        disease_code = get_disease_name(predicted_class)
        disease_name = format_disease_name(disease_code)
        
        all_predictions = []
        for i, conf in enumerate(predictions[0]):
            all_predictions.append({
                "disease_id": i,
                "disease_code": get_disease_name(i),
                "disease_name": format_disease_name(get_disease_name(i)),
                "confidence": float(conf)
            })
        
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        response = {
            "status": "success",
            "prediction": {
                "disease_id": int(predicted_class),
                "disease_code": disease_code,
                "disease_name": disease_name,
                "confidence": confidence,
                "confidence_percentage": f"{confidence*100:.2f}%"
            },
            "all_predictions": all_predictions[:3],
            "model_metadata": {
                "model_type": "MobileNetV2",
                "image_size": IMG_SIZE,
                "test_accuracy": metrics.get("test_accuracy") if metrics else None
            }
        }
        
        if disease_code == "Tomato___healthy":
            response["recommendation"] = "✓ Planta está saudável. Continue com manejo preventivo."
        else:
            response["recommendation"] = f"⚠️ Doença detectada: {disease_name}. Procure por tratamento adequado."
        
        return response
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erro ao processar imagem: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao processar predição: {str(e)}"
        )

@router.post("/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if vision_model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de visão não está carregado"
        )
    
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="Nenhum arquivo enviado"
        )
    
    results = []
    
    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            image_array = preprocess_image(contents)
            
            predictions = vision_model.predict(image_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            disease_code = get_disease_name(predicted_class)
            disease_name = format_disease_name(disease_code)
            
            results.append({
                "file_index": idx,
                "filename": file.filename,
                "status": "success",
                "prediction": {
                    "disease_id": int(predicted_class),
                    "disease_code": disease_code,
                    "disease_name": disease_name,
                    "confidence": confidence,
                    "confidence_percentage": f"{confidence*100:.2f}%"
                }
            })
        except Exception as e:
            results.append({
                "file_index": idx,
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "status": "success",
        "total_processed": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "results": results
    }

@router.get("/metrics")
async def get_model_metrics():
    if metrics is None:
        raise HTTPException(
            status_code=503,
            detail="Métricas não disponíveis"
        )
    
    return {
        "status": "success",
        "metrics": metrics
    }
