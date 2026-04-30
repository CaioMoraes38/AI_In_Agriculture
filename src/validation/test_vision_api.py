import requests
import json
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_vision_health():
    print("\n" + "="*70)
    print("🧪 TESTE 1: Health Check")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/vision/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def test_vision_root():
    print("\n" + "="*70)
    print("🧪 TESTE 2: Root Endpoint")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/vision/", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def test_vision_classes():
    print("\n" + "="*70)
    print("🧪 TESTE 3: Classes Endpoint")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/vision/classes", timeout=5)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Total de Classes: {data['total_classes']}")
        print(f"\nClasses disponíveis:")
        for cls in data['classes']:
            print(f"  {cls['id']}. {cls['name']} - {cls['description'][:50]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def test_vision_predict():
    print("\n" + "="*70)
    print("🧪 TESTE 4: Prediction Endpoint")
    print("="*70)
    
    test_image_dir = "plantvillagedataset/color/Tomato___healthy"
    
    if not os.path.exists(test_image_dir):
        print(f"⚠️  Diretório de teste não encontrado: {test_image_dir}")
        return False
    
    try:
        image_files = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png'))]
        
        if not image_files:
            print("⚠️  Nenhuma imagem encontrada")
            return False
        
        test_image_path = os.path.join(test_image_dir, image_files[0])
        
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/vision/predict", files=files, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"\nResponse:")
        print(f"  Status: {data['status']}")
        print(f"  Doença: {data['prediction']['disease_name']}")
        print(f"  Confiança: {data['prediction']['confidence_percentage']}")
        print(f"  Recomendação: {data['recommendation']}")
        
        print(f"\nTop 3 predições:")
        for pred in data['all_predictions'][:3]:
            print(f"  - {pred['disease_name']}: {pred['confidence']:.2%}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def test_vision_batch():
    print("\n" + "="*70)
    print("🧪 TESTE 5: Batch Prediction")
    print("="*70)
    
    test_image_dir = "plantvillagedataset/color/Tomato___healthy"
    
    if not os.path.exists(test_image_dir):
        print(f"⚠️  Diretório de teste não encontrado: {test_image_dir}")
        return False
    
    try:
        image_files = [f for f in os.listdir(test_image_dir) 
                      if f.endswith(('.jpg', '.png'))][:3]
        
        if not image_files:
            print("⚠️  Nenhuma imagem encontrada")
            return False
        
        files_list = []
        for image_file in image_files:
            image_path = os.path.join(test_image_dir, image_file)
            files_list.append(('files', (image_file, open(image_path, 'rb'))))
        
        response = requests.post(f"{BASE_URL}/vision/batch", files=files_list, timeout=30)
        
        for _, file_tuple in files_list:
            file_tuple[1].close()
        
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"\nResponse:")
        print(f"  Total processadas: {data['total_processed']}")
        print(f"  Sucesso: {data['successful']}")
        print(f"  Erros: {data['failed']}")
        
        print(f"\nResultados:")
        for result in data['results']:
            if result['status'] == 'success':
                print(f"  ✓ {result['filename']}: {result['prediction']['disease_name']} " +
                      f"({result['prediction']['confidence_percentage']})")
            else:
                print(f"  ❌ {result['filename']}: {result['error']}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def test_vision_metrics():
    print("\n" + "="*70)
    print("🧪 TESTE 6: Metrics Endpoint")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/vision/metrics", timeout=5)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data['metrics'], indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def test_api_root():
    print("\n" + "="*70)
    print("🧪 TESTE 7: API Root Endpoint")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Message: {data['message']}")
        print(f"Version: {data['version']}")
        print(f"Endpoints: {list(data['endpoints'].keys())}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def main():
    print("\n" + "🧪"*35)
    print("  TESTES DE API - MODELO DE VISÃO")
    print("🧪"*35)
    
    print("\n⚠️  Certifique-se de que a API está rodando em http://localhost:8000")
    print("   Execute: python src/api/main.py ou uvicorn src.api.main:app --reload\n")
    
    results = []
    
    results.append(("API Root", test_api_root()))
    results.append(("Vision Health", test_vision_health()))
    results.append(("Vision Root", test_vision_root()))
    results.append(("Classes", test_vision_classes()))
    results.append(("Single Prediction", test_vision_predict()))
    results.append(("Batch Prediction", test_vision_batch()))
    results.append(("Metrics", test_vision_metrics()))
    
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
        print("\n✅ TODOS OS TESTES DA API PASSARAM!")
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam.")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
