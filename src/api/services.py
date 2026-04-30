from pathlib import Path
import joblib
import numpy as np
from enum import Enum
from dtos import PredictionRequest, PredictionResponse


MODEL_PATH = Path(__file__).parent.parent / "models" / "modelsV1" / "irrigation_model_v1.pkl"

model_data = None


def load_model():
    global model_data
    try:
        if MODEL_PATH.exists():
            model_data = joblib.load(MODEL_PATH)
            print(f"✅ Modelo carregado com sucesso de: {MODEL_PATH}")
            print(f"   Features: {model_data.get('feature_names', [])}")
            return True
        else:
            print(f"⚠️ Modelo não encontrado em: {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_model_data():
    return model_data


def predict_irrigation(request: PredictionRequest) -> PredictionResponse:
    if model_data is None:
        raise ValueError("Modelo de ML não carregado. Verifique o arquivo do modelo.")
    
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    label_encoders = model_data.get('label_encoders', {})
    feature_names = model_data.get('feature_names', [])
    
    if not model:
        raise ValueError("Modelo não encontrado no arquivo carregado")
    
    if not feature_names:
        raise ValueError("Feature names não encontrados")
    
    data_dict = request.model_dump()
    
    for key, value in data_dict.items():
        if isinstance(value, Enum):
            data_dict[key] = value.value
    
    categorical_features = list(label_encoders.keys())
    numeric_features = [f for f in feature_names if f not in categorical_features]
    
    features_list = []
    
    for fname in feature_names:
        if fname not in data_dict:
            raise ValueError(f"Feature {fname} não encontrada nos dados de entrada")
        
        value = data_dict[fname]
        
        if fname in categorical_features:
            encoder = label_encoders[fname]
            try:
                encoded_value = encoder.transform([str(value)])[0]
                features_list.append(encoded_value)
            except Exception as e:
                raise ValueError(f"Erro ao codificar {fname}={value}: {str(e)}")
        else:
            features_list.append(float(value))
    
    features = np.array([features_list], dtype=np.float32)
    
    if scaler:
        numeric_indices = [feature_names.index(f) for f in numeric_features]
        numeric_values = features[:, numeric_indices]
        numeric_values_scaled = scaler.transform(numeric_values)
        features[:, numeric_indices] = numeric_values_scaled
    
    prediction = model.predict(features)[0]
    
    if prediction > 10:
        confidence = "Alta"
    elif prediction > 5:
        confidence = "Média"
    else:
        confidence = "Baixa"
    
    return PredictionResponse(
        status="success",
        prediction_mm=round(float(prediction), 2),
        confidence=confidence,
        message=f"Necessidade estimada de irrigação: {prediction:.2f} mm/dia"
    )
