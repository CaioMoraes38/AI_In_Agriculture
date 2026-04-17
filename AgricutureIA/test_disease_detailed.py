import base64
import requests
from PIL import Image, ImageDraw
from io import BytesIO
import random

print("=" * 60)
print("TESTE DE DIAGNÓSTICO - Modelo CNN Tomato")
print("=" * 60)

# Teste 1: Imagem toda branca (saudável)
print("\n[Teste 1] Imagem BRANCA (esperado: healthy)")
img = Image.new('RGB', (224, 224), color=(255, 255, 255))
buf = BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
response = requests.post(
    'http://localhost:8000/doencas/detectar',
    files={'arquivo': ('test.png', buf, 'image/png')}
)
if response.status_code == 200:
    data = response.json()
    print(f"  Resultado: {data['classe_prevista']}")
    print(f"  Confiança: {data['confianca_percentual']}%")
    print(f"  Top 3: {data['top_3_predicoes']}")
else:
    print(f"  Erro: {response.status_code}")

# Teste 2: Imagem com gradiente
print("\n[Teste 2] Imagem com GRADIENTE")
img = Image.new('RGB', (224, 224))
pixels = img.load()
for x in range(224):
    for y in range(224):
        val = int((x / 224) * 255)
        pixels[x, y] = (val, val, val)
buf = BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
response = requests.post(
    'http://localhost:8000/doencas/detectar',
    files={'arquivo': ('test.png', buf, 'image/png')}
)
if response.status_code == 200:
    data = response.json()
    print(f"  Resultado: {data['classe_prevista']}")
    print(f"  Confiança: {data['confianca_percentual']}%")
else:
    print(f"  Erro: {response.status_code}")

# Teste 3: Imagem com padrão radial
print("\n[Teste 3] Imagem com PADRÃO RADIAL")
img = Image.new('RGB', (224, 224))
pixels = img.load()
cx, cy = 112, 112
for x in range(224):
    for y in range(224):
        dist = ((x - cx)**2 + (y - cy)**2)**0.5
        val = int((dist / 158) * 255)
        pixels[x, y] = (val, val, val)
buf = BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
response = requests.post(
    'http://localhost:8000/doencas/detectar',
    files={'arquivo': ('test.png', buf, 'image/png')}
)
if response.status_code == 200:
    data = response.json()
    print(f"  Resultado: {data['classe_prevista']}")
    print(f"  Confiança: {data['confianca_percentual']}%")
else:
    print(f"  Erro: {response.status_code}")

# Teste 4: Imagem preta
print("\n[Teste 4] Imagem PRETA")
img = Image.new('RGB', (224, 224), color=(0, 0, 0))
buf = BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
response = requests.post(
    'http://localhost:8000/doencas/detectar',
    files={'arquivo': ('test.png', buf, 'image/png')}
)
if response.status_code == 200:
    data = response.json()
    print(f"  Resultado: {data['classe_prevista']}")
    print(f"  Confiança: {data['confianca_percentual']}%")
else:
    print(f"  Erro: {response.status_code}")

# Teste 5: Imagem com ruído
print("\n[Teste 5] Imagem com RUÍDO ALEATÓRIO")
img = Image.new('RGB', (224, 224))
pixels = img.load()
for x in range(224):
    for y in range(224):
        val = random.randint(0, 255)
        pixels[x, y] = (val, val, val)
buf = BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
response = requests.post(
    'http://localhost:8000/doencas/detectar',
    files={'arquivo': ('test.png', buf, 'image/png')}
)
if response.status_code == 200:
    data = response.json()
    print(f"  Resultado: {data['classe_prevista']}")
    print(f"  Confiança: {data['confianca_percentual']}%")
else:
    print(f"  Erro: {response.status_code}")

# Teste 6: Imagem colorida (verde)
print("\n[Teste 6] Imagem VERDE (verde = 255, outros = 100)")
img = Image.new('RGB', (224, 224), color=(100, 255, 100))
buf = BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
response = requests.post(
    'http://localhost:8000/doencas/detectar',
    files={'arquivo': ('test.png', buf, 'image/png')}
)
if response.status_code == 200:
    data = response.json()
    print(f"  Resultado: {data['classe_prevista']}")
    print(f"  Confiança: {data['confianca_percentual']}%")
else:
    print(f"  Erro: {response.status_code}")

# Teste 7: Imagem avermelhada
print("\n[Teste 7] Imagem VERMELHA (vermelho = 255, outros = 100)")
img = Image.new('RGB', (224, 224), color=(255, 100, 100))
buf = BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
response = requests.post(
    'http://localhost:8000/doencas/detectar',
    files={'arquivo': ('test.png', buf, 'image/png')}
)
if response.status_code == 200:
    data = response.json()
    print(f"  Resultado: {data['classe_prevista']}")
    print(f"  Confiança: {data['confianca_percentual']}%")
else:
    print(f"  Erro: {response.status_code}")

print("\n" + "=" * 60)
print("ANÁLISE: O modelo está generalizando demais?")
print("Se tudo retorna 'healthy', o modelo pode estar com problema")
print("=" * 60)
