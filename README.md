# 🌾 Sistema de Previsão de Irrigação com ML

> **Status:** ✅ Pronto para uso | **Acurácia:** 98.70% | **Modelo:** Random Forest

## 📋 Índice

1. [Início Rápido](#-início-rápido)
2. [Instalação](#-instalação)
3. [Estrutura do Projeto](#-estrutura-do-projeto)
4. [Como Usar](#-como-usar)
5. [Exemplos](#-exemplos)
6. [API Endpoints](#-api-endpoints)
7. [Documentação Detalhada](#-documentação-detalhada)

---

## 🚀 Início Rápido

### 1️⃣ Instalar Dependências
```bash
cd "c:\TCC - Code\IA - TCC\AgricutureIA"
pip install -r src/requirements.txt
```

### 2️⃣ Treinar o Modelo
```bash
cd src
python scripts/treinar_irrigacao.py
```

Resultado: **98.70% acurácia** ✅

### 3️⃣ Iniciar a API
```bash
python main.py
```

### 4️⃣ Acessar Documentação
Abra no navegador: **http://localhost:8000/docs**

---

## ⚙️ Instalação

### Requisitos
- Python 3.8+
- pip

### Passos

1. **Clone/Acesse o projeto:**
```bash
cd "c:\TCC - Code\IA - TCC\AgricutureIA"
```

2. **Instale as dependências:**
```bash
pip install -r src/requirements.txt
```

**Pacotes instalados:**
- `fastapi` - API REST
- `scikit-learn` - Machine Learning
- `pandas` - Processamento de dados
- `numpy` - Cálculos numéricos
- `uvicorn` - Servidor web

3. **Pronto!** ✅

---

## 📁 Estrutura do Projeto

```
AgricutureIA/
│
├── src/
│   ├── main.py                          # Aplicação principal (FastAPI)
│   ├── config.py                        # Configurações globais
│   ├── requirements.txt                 # Dependências Python
│   │
│   ├── scripts/                         # Scripts de utilidade
│   │   ├── treinar_irrigacao.py         # Treinar modelo ML
│   │   └── test_irrigacao.py            # Teste interativo
│   │
│   ├── docs/                            # Documentação
│   │   ├── INICIO_RAPIDO.md             # Este arquivo
│   │   ├── GUIA_COMPLETO.md             # Guia detalhado
│   │   ├── API_ENDPOINTS.md             # Referência de endpoints
│   │   └── TECNICO.md                   # Detalhes técnicos
│   │
│   ├── models/                          # Modelos treinados
│   │   ├── irrigation_model.pkl         # Modelo Random Forest
│   │   ├── irrigation_scaler.pkl        # Normalizador
│   │   ├── irrigation_encoders.json     # Codificadores
│   │   ├── irrigation_features.json     # Lista de features
│   │   └── irrigation_model_info.json   # Metadados
│   │
│   ├── ia/                              # Módulo de IA
│   │   ├── services/
│   │   │   ├── irrigation_ml_service.py     # ⭐ Serviço ML principal
│   │   │   ├── irrigation_service.py        # Serviço simples (fallback)
│   │   │   └── vision_service.py            # Serviço de visão
│   │   └── ...
│   │
│   └── api/                             # API REST
│       ├── routes/
│       │   ├── irrigation.py            # ⭐ Endpoints de irrigação
│       │   └── vision.py                # Endpoints de visão
│       ├── schemas/                     # Modelos Pydantic
│       └── dependencies.py              # Injeção de dependências
│
├── Irrigation data/
│   └── irrigation_prediction.csv        # Dataset de treino (10.000 registros)
│
└── BaseImage/                           # Base de imagens
    └── plantvillage dataset/
```

---

## 💡 Como Usar

### Opção A: Teste Interativo

```bash
cd src
python scripts/test_irrigacao.py
```

**Menu interativo com:**
- Treinar modelo
- Testar com 3 variáveis
- Testar com dados completos
- Ver exemplos

### Opção B: API REST

```bash
cd src
python main.py
```

Acesse: http://localhost:8000/docs

### Opção C: Python Direto

```python
from ia.services.irrigation_ml_service import IrrigationMLService

service = IrrigationMLService()

# Previsão simples
resultado = service.prever_irrigacao(
    temperatura=28.5,
    umidade=65.0,
    chuva=15.0
)

print(f"Classe: {resultado['classe_prevista']}")
print(f"Volume: {resultado['volume_litros']}L")
print(f"Confiança: {resultado['confianca']}%")
```

---

## 📊 Exemplos

### Exemplo 1: Previsão Simples (API)

```bash
curl "http://localhost:8000/irrigacao/ml/prever?temperatura=28.5&umidade=65&chuva=15"
```

**Resposta:**
```json
{
  "classe_prevista": "Medium",
  "confianca": 82.45,
  "volume_litros": 17.5,
  "volume_base": 15.0,
  "ajustes_aplicados": [
    {"fator": "Temperatura alta", "ajuste": "+2.5L"}
  ],
  "nivel_alerta": "normal",
  "modelo_usado": "random_forest_ml"
}
```

### Exemplo 2: Previsão Completa (Maior Precisão)

```bash
curl "http://localhost:8000/irrigacao/ml/prever?temperatura=28.5&umidade=65&chuva=15&soil_type=Clay&crop_type=Wheat&crop_growth_stage=Vegetative"
```

**Resultado:** 91%+ confiança

### Exemplo 3: Cenários Especiais

#### 🔥 Emergência (Muito quente e seco)
```bash
curl "http://localhost:8000/irrigacao/ml/prever?temperatura=38&umidade=35&chuva=0"
# Resultado: High - 30-35L - Alerta: crítico
```

#### 🌧️ Com Chuva
```bash
curl "http://localhost:8000/irrigacao/ml/prever?temperatura=22&umidade=75&chuva=80"
# Resultado: Low - 5L - Alerta: chuva_prevista
```

---

## 🌐 API Endpoints

### Previsão de Irrigação

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| **GET/POST** | `/irrigacao/ml/prever` | Fazer previsão com dados |
| **GET** | `/irrigacao/ml/info` | Info do modelo |
| **GET** | `/irrigacao/ml/parametros` | Variáveis opcionais |
| **POST** | `/irrigacao/ml/exemplo` | Exemplo de previsão |

### Endpoints Legados

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| **POST** | `/irrigacao/calcular` | Cálculo simples (sem ML) |
| **GET** | `/irrigacao/saude` | Status do serviço |

---

## 📚 Documentação Detalhada

| Documento | Conteúdo |
|-----------|----------|
| [GUIA_COMPLETO.md](docs/GUIA_COMPLETO.md) | Guia passo a passo (mais detalhado) |
| [API_ENDPOINTS.md](docs/API_ENDPOINTS.md) | Referência completa de endpoints |
| [TECNICO.md](docs/TECNICO.md) | Detalhes técnicos e arquitetura |

---

## 🎯 Variáveis Aceitas

### Obrigatórias (3)
```
temperatura        float   -50 a 60 °C
umidade           float   0 a 100 %
chuva             float   0 a 200+ mm
```

### Opcionais (Melhoram Precisão)
```
soil_type              Clay / Silt / Sandy / Loamy
crop_type              Wheat / Maize / Rice / Cotton / Potato / Sugarcane
crop_growth_stage      Sowing / Vegetative / Flowering / Harvest
season                 Rabi / Kharif / Zaid
soil_moisture          0 a 100 %
soil_ph                4 a 8
sunlight_hours         0 a 12
wind_speed             0 a 20 km/h
irrigation_type        Rainfed / Canal / Sprinkler / Drip
water_source           Reservoir / Groundwater / River / Rainwater
region                 North / South / East / West / Central
mulching_used          Yes / No
```

---

## 📊 Saídas de Irrigação

| Classe | Volume Base | Uso |
|--------|------------|-----|
| **Low** | 5L | Pouca necessidade (chuva/solo úmido) |
| **Medium** | 15L | Necessidade normal |
| **High** | 25L | Muita necessidade (seco/quente) |

*+ ajustes automáticos baseados nas condições*

---

## 🔄 Retreinar com Novos Dados

1. **Adicione dados** ao `Irrigation data/irrigation_prediction.csv`
2. **Execute:** `python scripts/treinar_irrigacao.py`
3. **API carrega automaticamente** o novo modelo

```bash
cd src
python scripts/treinar_irrigacao.py
```

---

## 🔧 Troubleshooting

### ❌ "Módulo não encontrado"
```bash
pip install -r requirements.txt
```

### ❌ "Porta 8000 já em uso"
```bash
uvicorn main:app --port 8001
```

### ❌ "Modelo não carregado"
```bash
python scripts/treinar_irrigacao.py
```

---

## 📈 Performance

| Métrica | Valor |
|---------|-------|
| Acurácia | 98.70% |
| Tempo de Treinamento | ~5 segundos |
| Tempo de Predição | <100ms |
| Features | 19 |
| Treino (samples) | 8.000 |
| Teste (samples) | 2.000 |

---

## 🤝 Integração

### Com seu App Web
```javascript
const response = await fetch(
  'http://localhost:8000/irrigacao/ml/prever?temperatura=28&umidade=65&chuva=15'
);
const data = await response.json();
console.log(`Irriga com ${data.volume_litros}L`);
```

### Com seu App Mobile
```kotlin
val response = HttpClient().get<IrrigacaoResponse>(
    "http://api.seu-dominio.com/irrigacao/ml/prever?temperatura=28&umidade=65&chuva=15"
)
```

### Com IoT/Sensores
```python
import requests

sensor_data = {
    'temperatura': 28.5,
    'umidade': 65.0,
    'chuva': 15.0
}

response = requests.get(
    'http://seu-servidor:8000/irrigacao/ml/prever',
    params=sensor_data
)

volume = response.json()['volume_litros']
# Ativar bomba de irrigação...
```

---

## ℹ️ Informações do Modelo

**Tipo:** Random Forest Classifier  
**Acurácia:** 98.70%  
**Classes:** Low, Medium, High  
**Features:** 19 (temperatura, umidade, chuva, solo, cultura, etc.)  
**Treinado em:** 10.000 registros  

**Features Mais Importantes:**
1. Umidade do Solo (21.41%)
2. Estágio de Crescimento (15.08%)
3. Precipitação (14.66%)
4. Temperatura (12.22%)
5. Velocidade do Vento (10.84%)

---

##  Próximos Passos

- [ ] Treinar modelo → `python scripts/treinar_irrigacao.py`
- [ ] Iniciar API → `python main.py`
- [ ] Testar endpoints → Acesso http://localhost:8000/docs
- [ ] Integrar com seu sistema
- [ ] Adicionar dados novos periodicamente
- [ ] Retreinar modelo mensalmente

---

## 📝 Licença

Projeto educacional - TCC Agricultura com IA

---

**Pronto para usar!** 🎉 Dúvidas? Veja a [documentação detalhada](docs/GUIA_COMPLETO.md).
