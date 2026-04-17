import base64
import requests
from PIL import Image
from io import BytesIO
import random

# Criar 3 imagens diferentes
for i in range(3):
    # Imagem com padrão diferente
    arr = [[random.randint(0, 255) for _ in range(224)] for _ in range(224)]
    img = Image.new('RGB', (224, 224))
    pixels = img.load()
    for x in range(224):
        for y in range(224):
            pixels[x, y] = (arr[y][x], arr[y][x], arr[y][x])
    
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    # Upload
    response = requests.post(
        'http://localhost:8000/doencas/detectar',
        files={'arquivo': ('test.png', buf, 'image/png')}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Teste {i+1}: {data['classe_prevista']} ({data['confianca_percentual']}%)")
    else:
        print(f"Erro: {response.status_code}")
