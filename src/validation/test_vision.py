import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import pickle

sys.path.append('src')

MODEL_PATH = "src/models/modelVisionV1/vision_model_v1.h5"
METRICS_PATH = "src/models/modelVisionV1/vision_model_metrics.json"
LABELS_PATH = "src/models/modelVisionV1/class_labels.pkl"
TEST_IMAGE_PATH = "plantvillagedataset/color"

IMG_SIZE = 224

def test_model_loading():
    print("\n" + "="*70)
    print("🧪 TESTE 1: Carregamento do Modelo")
    print("="*70)
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Modelo carregado com sucesso")
        print(f"  - Tipo: CNN MobileNetV2")
        print(f"  - Entrada: {model.input_shape}")
        print(f"  - Saída: {model.output_shape}")
        return True, model
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {str(e)}")
        return False, None

def test_metrics_loading():
    print("\n" + "="*70)
    print("🧪 TESTE 2: Carregamento de Métricas")
    print("="*70)
    
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        
        print("✓ Métricas carregadas com sucesso")
        print(f"  - Acurácia de Treino: {metrics.get('train_accuracy', 'N/A'):.4f}")
        print(f"  - Acurácia de Teste: {metrics.get('test_accuracy', 'N/A'):.4f}")
        print(f"  - Tipo de Modelo: {metrics.get('model_type', 'N/A')}")
        print(f"  - Número de Classes: {metrics.get('num_classes', 'N/A')}")
        return True, metrics
    except Exception as e:
        print(f"❌ Erro ao carregar métricas: {str(e)}")
        return False, None

def test_labels_loading():
    print("\n" + "="*70)
    print("🧪 TESTE 3: Carregamento de Labels")
    print("="*70)
    
    try:
        with open(LABELS_PATH, 'rb') as f:
            labels = pickle.load(f)
        
        print("✓ Labels carregados com sucesso")
        print(f"  - Número de classes: {len(labels)}")
        print(f"  - Classes: {list(labels.keys())}")
        return True, labels
    except Exception as e:
        print(f"❌ Erro ao carregar labels: {str(e)}")
        return False, None

def test_image_preprocessing():
    print("\n" + "="*70)
    print("🧪 TESTE 4: Preprocessamento de Imagem")
    print("="*70)
    
    test_image_dir = os.path.join(TEST_IMAGE_PATH, "Tomato___healthy")
    
    if not os.path.exists(test_image_dir):
        print(f"⚠️  Diretório de teste não encontrado: {test_image_dir}")
        return False
    
    try:
        image_files = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png'))]
        
        if not image_files:
            print("⚠️  Nenhuma imagem encontrada no diretório de teste")
            return False
        
        test_image_path = os.path.join(test_image_dir, image_files[0])
        
        image = Image.open(test_image_path).convert('RGB')
        original_shape = image.size
        
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        print("✓ Imagem preprocessada com sucesso")
        print(f"  - Caminho: {test_image_path}")
        print(f"  - Tamanho original: {original_shape}")
        print(f"  - Tamanho redimensionado: {image.size}")
        print(f"  - Shape do array: {image_array.shape}")
        print(f"  - Range de valores: [{image_array.min():.4f}, {image_array.max():.4f}]")
        
        return True, image_array
    except Exception as e:
        print(f"❌ Erro ao preprocessar imagem: {str(e)}")
        return False, None

def test_model_prediction(model, image_array):
    print("\n" + "="*70)
    print("🧪 TESTE 5: Predição do Modelo")
    print("="*70)
    
    try:
        predictions = model.predict(image_array, verbose=0)
        
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
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
        
        predicted_disease = disease_map[predicted_class]
        
        print("✓ Predição realizada com sucesso")
        print(f"  - Classe predita: {predicted_class}")
        print(f"  - Doença: {predicted_disease}")
        print(f"  - Confiança: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"\n  Top 3 predições:")
        
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        for i, idx in enumerate(top_3_idx, 1):
            conf = predictions[0][idx]
            disease = disease_map[idx]
            print(f"    {i}. {disease}: {conf:.4f} ({conf*100:.2f}%)")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao fazer predição: {str(e)}")
        return False

def test_model_architecture(model):
    print("\n" + "="*70)
    print("🧪 TESTE 6: Arquitetura do Modelo")
    print("="*70)
    
    try:
        print("✓ Arquitetura do modelo:")
        model.summary()
        
        total_params = model.count_params()
        trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\n  Resumo de parâmetros:")
        print(f"  - Total: {total_params:,}")
        print(f"  - Treináveis: {trainable_params:,}")
        print(f"  - Não treináveis: {non_trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao analisar arquitetura: {str(e)}")
        return False

def test_batch_prediction(model):
    print("\n" + "="*70)
    print("🧪 TESTE 7: Predição em Lote")
    print("="*70)
    
    try:
        batch_size = 4
        dummy_batch = np.random.rand(batch_size, IMG_SIZE, IMG_SIZE, 3)
        
        predictions = model.predict(dummy_batch, verbose=0)
        
        print("✓ Predição em lote realizada com sucesso")
        print(f"  - Tamanho do lote: {batch_size}")
        print(f"  - Shape de entrada: {dummy_batch.shape}")
        print(f"  - Shape de saída: {predictions.shape}")
        print(f"  - Tempo de predição: OK")
        
        return True
    except Exception as e:
        print(f"❌ Erro na predição em lote: {str(e)}")
        return False

def test_inference_speed(model):
    print("\n" + "="*70)
    print("🧪 TESTE 8: Velocidade de Inferência")
    print("="*70)
    
    try:
        import time
        
        dummy_image = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3)
        
        start_time = time.time()
        for _ in range(10):
            _ = model.predict(dummy_image, verbose=0)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        print("✓ Testes de velocidade concluídos")
        print(f"  - Número de predições: 10")
        print(f"  - Tempo total: {end_time - start_time:.4f}s")
        print(f"  - Tempo médio por predição: {avg_time:.4f}s")
        print(f"  - Predições por segundo: {1/avg_time:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao testar velocidade: {str(e)}")
        return False

def main():
    print("\n" + "🧪"*35)
    print("  TESTES DO MODELO DE VISÃO - DETECÇÃO DE DOENÇAS")
    print("🧪"*35)
    
    results = []
    
    model_loaded, model = test_model_loading()
    results.append(("Carregamento do Modelo", model_loaded))
    
    metrics_loaded, metrics = test_metrics_loading()
    results.append(("Carregamento de Métricas", metrics_loaded))
    
    labels_loaded, labels = test_labels_loading()
    results.append(("Carregamento de Labels", labels_loaded))
    
    image_preprocessed, image_array = test_image_preprocessing()
    results.append(("Preprocessamento de Imagem", image_preprocessed))
    
    if model_loaded and image_preprocessed:
        prediction_ok = test_model_prediction(model, image_array)
        results.append(("Predição do Modelo", prediction_ok))
    
    if model_loaded:
        architecture_ok = test_model_architecture(model)
        results.append(("Arquitetura do Modelo", architecture_ok))
        
        batch_ok = test_batch_prediction(model)
        results.append(("Predição em Lote", batch_ok))
        
        speed_ok = test_inference_speed(model)
        results.append(("Velocidade de Inferência", speed_ok))
    
    print("\n" + "="*70)
    print("📊 RESUMO DOS TESTES")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSOU" if result else "❌ FALHOU"
        print(f"{status} - {test_name}")
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n✅ TODOS OS TESTES PASSARAM COM SUCESSO!")
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam. Verifique os erros acima.")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
