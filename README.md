# 🌾 AgriculturaIA - Predição Inteligente de Irrigação e Análise de Doenças em Plantas

![Status](https://img.shields.io/badge/Status-Produção%20Pronto-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14.0-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-1.0-orange)

> **Sistema completo de IA para Agricultura 5.0**: Integra predição inteligente de necessidade de irrigação usando Machine Learning, classificação de doenças em plantas via Deep Learning, processamento avançado de dados agrícolas com cálculos físicos de evapotranspiração (Penman-Monteith), e uma API REST moderna para integração com sistemas agrícolas.

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Tratamento de Dados](#tratamento-de-dados)
- [Modelos de IA](#modelos-de-ia)
- [Estrutura de Pastas](#estrutura-de-pastas)
- [Instalação e Uso](#instalação-e-uso)
- [API REST](#api-rest)
- [Performance](#performance)
- [Gráficos e Análises](#gráficos-e-análises)

---

## 🎯 Visão Geral

AgriculturaIA é uma solução completa que integra:

✅ **Predição de Irrigação**: Machine Learning para otimizar o uso de água em lavouras
✅ **Análise de Doenças**: Deep Learning para identificar doenças em plantas
✅ **API REST**: Interface moderna e escalável para integração com sistemas agrícolas
✅ **Processamento de Dados Avançado**: Cálculos agrofísicos e normalização de dados
✅ **Validação Científica**: Implementação da fórmula Penman-Monteith para evapotranspiração

---

## 🏗️ Arquitetura do Projeto

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND / APLICAÇÕES                │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │   API REST      │
        │  (FastAPI)      │
        └────────┬────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼──┐    ┌────▼────┐  ┌───▼─────┐
│Models│    │Services │  │ DTOs    │
│  ML  │    │ & Logic │  │Validation
└───┬──┘    └────┬────┘  └───┬─────┘
    │            │            │
┌───▼──────────────────────────▼────┐
│     Core Processing Layer         │
│  • Feature Engineering            │
│  • Data Normalization             │
│  • Label Encoding                 │
│  • Validation                     │
└───┬─────────────────────────────┬─┘
    │                             │
┌───▼────────────────┐  ┌────────▼──────────────┐
│ Irrigation Model   │  │  Vision Model         │
│ (Random Forest)    │  │ (CNN/TensorFlow)     │
└────────────────────┘  └───────────────────────┘
```

---

## 💻 Tecnologias Utilizadas

### 🔤 Linguagens
| Linguagem | Versão | Utilização |
|-----------|--------|-----------|
| **Python** | 3.x | Linguagem principal do projeto |
| **Markdown** | - | Documentação |

### 🌐 Backend & APIs
| Tecnologia | Versão | Função |
|-----------|--------|--------|
| **FastAPI** | 0.104.1 | Framework web de alta performance para criar APIs REST |
| **Uvicorn** | 0.24.0 | Servidor ASGI para rodar a aplicação |
| **Pydantic** | 2.5.0 | Validação de dados e DTOs |

### 🤖 Machine Learning & IA
| Biblioteca | Versão | Funcionalidade |
|-----------|--------|--------|
| **scikit-learn** | 1.3.2 | RandomForest, StandardScaler, LabelEncoder, Métricas |
| **XGBoost** | 2.0.2 | Gradient Boosting (modelo alternativo) |
| **TensorFlow** | 2.14.0 | Redes Neurais Convolucionais para classificação |
| **joblib** | 1.3.2 | Serialização e carregamento de modelos |

### 📊 Processamento de Dados
| Biblioteca | Versão | Função |
|-----------|--------|--------|
| **pandas** | 2.1.3 | Manipulação de DataFrames e CSVs |
| **NumPy** | 1.26.2 | Operações numéricas e cálculos científicos |

### 📈 Visualização
| Biblioteca | Versão | Gráficos |
|-----------|--------|---------|
| **Matplotlib** | 3.8.2 | Gráficos estáticos e análises |
| **Seaborn** | 0.13.0 | Visualizações estatísticas avançadas |

### 🖼️ Processamento de Imagens
| Biblioteca | Versão | Funcionalidade |
|-----------|--------|--------|
| **OpenCV** | 4.8.1.78 | Processamento de imagens |
| **Pillow** | 10.1.0 | Manipulação de imagens |

---

## 🔄 Tratamento de Dados

### 📥 Pipeline de Dados

#### 1️⃣ **Ingestão e Limpeza**
```python
✓ Carregamento de arquivos CSV
✓ Remoção de espaços em branco em colunas
✓ Sanitização de valores de texto
✓ Detecção e tratamento de valores ausentes
✓ Validação de ranges de dados
```

#### 2️⃣ **Normalização e Codificação**

**Normalização Numérica (StandardScaler)**
```
Z-score: (x - μ) / σ
- Média: 0
- Desvio Padrão: 1
- Aplicado a: Temperatura, Umidade, Vento, Insolação, etc.
```

**Codificação Categórica (LabelEncoder)**
```
Variáveis Codificadas:
├── Crop_Type (Wheat, Maize, Cotton, Rice, Sugarcane, Potato)
├── Soil_Type (Clay, Sandy, Loam)
├── Growth_Stage (Initial, Vegetative, Mid, Late)
├── Irrigation_Type (Rainfed, Canal, Drip, Sprinkler)
├── Water_Source (Reservoir, Groundwater, River, Rainwater)
├── Mulching_Used (Yes, No)
└── Season (Rabi, Kharif, Zaid)
```

#### 3️⃣ **Feature Engineering**

**11 Features Selecionadas:**
```
Variáveis Climáticas:
  1. Temperature_C (°C)
  2. Humidity (%)
  3. Wind_Speed_kmh (km/h)
  4. Sunlight_Hours (horas/dia)
  5. Rainfall_mm (mm)

Características do Solo:
  6. Soil_Type
  7. Soil_Moisture (%)

Características da Cultura:
  8. Crop_Type
  9. Crop_Growth_Stage
  10. Field_Area_hectare

Características da Irrigação:
  11. Irrigation_Type
```

#### 4️⃣ **Cálculo de Evapotranspiração (ETc)**

**Fórmula Penman-Monteith FAO-56:**
```
ETc = ET₀ × Kc

Componentes Calculados:
├── Pressão Atmosférica (barometria)
├── Pressão de Saturação do Vapor (Magnus)
├── Pressão Atual do Vapor
├── Radiação Extraterrestre
├── Radiação Solar
├── Radiação Líquida de Onda Curta
├── Radiação Líquida de Onda Longa
└── Coeficiente de Cultura (Kc)
```

**Variáveis de Entrada:**
- Localização: Sorocaba, SP (-23.5018°, -47.4711°)
- Elevação: 580 m
- Coordenadas precisas para cálculos agrofísicos

#### 5️⃣ **Divisão Treino/Teste**
```
Total de Registros: 10.000
├── Treino: 8.000 (80%)
├── Teste: 2.000 (20%)
└── Validação Cruzada: K-Fold
```

---

## 🧠 Modelos de IA

### 🏆 Modelo Principal: Random Forest Regressor

**Tipo:** Ensemble Learning - Regressão
**Origem:** scikit-learn
**Objetivo:** Predição de necessidade de irrigação (ETc em mm/dia)

**Arquitetura:**
```
Random Forest
├── 100 Árvores de Decisão
├── Max Depth: 20
├── Min Samples Split: 5
├── Features: 11 variáveis
└── Parallelization: Multi-threading
```

**Performance:**

| Métrica | Treino | Teste | CV Média |
|---------|--------|-------|----------|
| **R² Score** | 99.63% | 98.15% | 98.34% |
| **MAE (mm/dia)** | 12.36 | 27.84 | - |
| **RMSE (mm/dia)** | 16.88 | 37.80 | - |
| **MAPE** | 1.43% | 3.23% | - |

**Acurácia por Faixa de ETc:**

| Faixa de ETc | Acurácia | Quantidade |
|-------------|----------|-----------|
| Muito Baixa (0-500) | 92.53% | 80 |
| Baixa (500-1000) | 96.75% | 1.239 |
| Média (1000-1500) | 97.34% | 612 |
| Alta (1500-2000) | 97.06% | 69 |

**Feature Importance (Importância de Características):**

```
1. Sunlight_Hours          36.1% ████████████████████████████░░
2. Crop_Type              25.7% ██████████████████░░░░░░░░░░░░
3. Temperature_C          24.0% █████████████████░░░░░░░░░░░░░
4. Wind_Speed_kmh         12.1% █████████░░░░░░░░░░░░░░░░░░░░░
5. Demais Features        < 2%  
```

---

### 🔍 Modelo de Visão: CNN (Convolutional Neural Network)

**Tipo:** Deep Learning - Classificação de Imagens
**Origem:** TensorFlow/Keras
**Objetivo:** Identificar doenças em plantas de tomate

**Dataset:**
```
Plant Village Dataset
├── Imagens: Coloridas (RGB)
├── Classes: 10 doenças de tomate
└── Tamanho: 256x256 pixels
```

**Categorias Detectadas:**
1. 🍅 Tomato___Healthy (Saudável)
2. 🔴 Tomato___Bacterial_spot (Mancha Bacteriana)
3. 🟤 Tomato___Early_blight (Requeima Precoce)
4. ⚫ Tomato___Late_blight (Requeima Tardia)
5. 🟢 Tomato___Leaf_Mold (Mofo da Folha)
6. 🟠 Tomato___Septoria_leaf_spot (Mancha Septória)
7. 🕷️ Tomato___Spider_mites (Ácaro Rajado)
8. 🎯 Tomato___Target_Spot (Mancha Alvo)
9. 🦠 Tomato___Tomato_mosaic_virus (Vírus do Mosaico)
10. 🟡 Tomato___Tomato_Yellow_Leaf_Curl_Virus (Vírus do Enrolamento)

---

### 📊 Métricas de Avaliação

**Para Regressão (Modelo de Irrigação):**

| Métrica | Fórmula | Interpretação |
|---------|---------|----------------|
| **R² Score** | 1 - (SS_residual / SS_total) | Percentual de variância explicada (0-100%) |
| **MAE** | Σ\|y_real - y_pred\| / n | Erro absoluto médio |
| **RMSE** | √(Σ(y_real - y_pred)² / n) | Raiz do erro quadrático médio |
| **MAPE** | 100 × Σ\|y_real - y_pred\| / y_real / n | Erro percentual absoluto médio |

**Para Classificação (Modelo de Doenças):**
```
- Acurácia (Accuracy)
- Precisão (Precision)
- Recall (Sensibilidade)
- F1-Score
- Matriz de Confusão
```

---

## 📁 Estrutura de Pastas

```
AgriculturaIA/
│
├── 📄 README.md                        ← Você está aqui!
├── 📄 requirements.txt                 ← Dependências Python
├── 📄 TECNOLOGIAS_E_BIBLIOTECAS.md    ← Documentação técnica
│
├── 📁 src/
│   │
│   ├── 🌐 api/
│   │   ├── __init__.py
│   │   ├── main.py                    ← Aplicação FastAPI
│   │   ├── services.py                ← Lógica de negócio
│   │   ├── routeIrrigationApi.py      ← Endpoints de irrigação
│   │   ├── routeVisionApi.py          ← Endpoints de visão
│   │   │
│   │   └── 📁 dtos/                   ← Data Transfer Objects
│   │       ├── models.py              ← Pydantic models
│   │       └── schemas.py             ← Enums e esquemas
│   │
│   ├── 🧹 preprocessing/
│   │   ├── evapotranspirationPenmanMonteith.py  ← Cálculos agrofísicos
│   │   ├── irrigation_prediction.csv  ← Dataset bruto
│   │   └── irrigation_prediction_with_etc.csv   ← Dataset processado
│   │
│   ├── 📊 graphics/
│   │   ├── gerar_graficos_modelo.py   ← Visualizações
│   │   ├── 01_feature_importance.png
│   │   ├── 02_metricas_desempenho.png
│   │   ├── 03_acuracia_por_faixa.png
│   │   ├── 04_distribuicao_erros.png
│   │   ├── 05_predicoes_vs_reais.png
│   │   ├── 06_sensibilidade_features.png
│   │   ├── 07_analise_dados_treinamento.png
│   │   └── 08_resumo_modelo.png
│   │
│   ├── 🎓 training/
│   │   │
│   │   ├── 🚰 irrigation/
│   │   │   ├── trainingIrrigation.py   ← Treinamento do modelo
│   │   │   └── validacao_resultados.py ← Validação e testes
│   │   │
│   │   └── 👁️ vision/
│   │       ├── trainingVision.py       ← Treinamento CNN
│   │       └── validacao_imagens.py    ← Validação de imagens
│   │
│   ├── 🎯 models/
│   │   └── 📁 modelsV1/
│   │       ├── irrigation_model_v1.pkl ← Modelo treinado (Random Forest)
│   │       ├── label_encoders.pkl      ← Codificadores categóricos
│   │       └── scaler.pkl              ← StandardScaler treinado
│   │
│   └── ✔️ validation/
│       ├── test_irrigation.py
│       ├── test_vision.py
│       └── test_api.py
│
├── 📁 plantvillagedataset/
│   └── 🖼️ color/
│       ├── Tomato___Bacterial_spot/
│       ├── Tomato___Early_blight/
│       ├── Tomato___healthy/
│       ├── Tomato___Late_blight/
│       ├── Tomato___Leaf_Mold/
│       ├── Tomato___Septoria_leaf_spot/
│       ├── Tomato___Spider_mites Two-spotted_spider_mite/
│       ├── Tomato___Target_Spot/
│       ├── Tomato___Tomato_mosaic_virus/
│       └── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│
├── 📁 Irrigation/
│   └── irrigation_prediction.csv       ← Dataset de irrigação
│
└── 🐍 venv/                           ← Virtual environment
```

---

## 🚀 Instalação e Uso

### 1️⃣ Pré-requisitos
```bash
✓ Python 3.8 ou superior
✓ pip (gerenciador de pacotes)
✓ Virtualenv (recomendado)
```

### 2️⃣ Clone e Configuração
```bash
# Clone o repositório
git clone <seu-repositorio>
cd AgriculturaIA

# Crie ambiente virtual
python -m venv venv

# Ative o ambiente
# No Windows:
.\venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
```

### 3️⃣ Instale Dependências
```bash
pip install -r requirements.txt
```

### 4️⃣ Processar Dados
```bash
# Calcular ETc com Penman-Monteith
cd src/preprocessing
python evapotranspirationPenmanMonteith.py
```

### 5️⃣ Treinar Modelos
```bash
# Treinar modelo de irrigação
cd src/training/irrigation
python trainingIrrigation.py

# Treinar modelo de visão (opcional)
cd ../vision
python trainingVision.py
```

### 6️⃣ Gerar Gráficos de Análise
```bash
cd src/graphics
python gerar_graficos_modelo.py
```

### 7️⃣ Iniciar API REST
```bash
cd src/api
python main.py

# Ou com uvicorn diretamente:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🌐 API REST

### 📍 Endpoints Disponíveis

#### 1. **Health Check**
```bash
GET /irrigation/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

#### 2. **Predição de Irrigação**
```bash
POST /irrigation/predict
```

**Corpo da Requisição:**
```json
{
  "Soil_Type": "Clay",
  "Soil_pH": 7.2,
  "Soil_Moisture": 45.0,
  "Organic_Carbon": 1.5,
  "Electrical_Conductivity": 0.8,
  "Temperature_C": 28.5,
  "Humidity": 65.0,
  "Rainfall_mm": 0.0,
  "Sunlight_Hours": 8.5,
  "Wind_Speed_kmh": 12.0,
  "Crop_Type": "Maize",
  "Crop_Growth_Stage": "Vegetative",
  "Season": "Kharif",
  "Irrigation_Type": "Drip",
  "Water_Source": "Groundwater",
  "Field_Area_hectare": 5.0,
  "Mulching_Used": "Yes",
  "Previous_Irrigation_mm": 50.0,
  "Region": 1
}
```

**Resposta de Sucesso (200):**
```json
{
  "status": "success",
  "prediction_mm": 145.32,
  "confidence": "Alta",
  "message": "Necessidade estimada de irrigação: 145.32 mm/dia"
}
```

---

#### 3. **Análise de Doenças em Plantas**
```bash
POST /vision/predict
Content-Type: multipart/form-data

[arquivo de imagem JPEG/PNG]
```

**Resposta:**
```json
{
  "status": "success",
  "disease": "Tomato___Early_blight",
  "confidence": 0.94,
  "message": "Requeima Precoce detectada com 94% de confiança"
}
```

---

### 📚 Documentação Interativa

Acesse a documentação Swagger:
```
http://localhost:8000/docs
```

---

## 📊 Performance

### 🎯 Modelo de Irrigação

| Métrica | Valor |
|---------|-------|
| **Precisão (R²)** | 98.15% |
| **Erro Médio (MAE)** | 27.84 mm/dia |
| **RMSE** | ~37.80 mm/dia |
| **Acurácia <10%** | 97.9% |
| **Acurácia <5%** | 88.7% |
| **Reproducibilidade** | 100% |

### 🖼️ Modelo de Visão

| Métrica | Valor |
|---------|-------|
| **Acurácia** | ~95%+ |
| **Classes** | 10 doenças |
| **Dataset** | 10.000+ imagens |
| **Entrada** | 256x256 RGB |

---

## 📈 Gráficos e Análises

Todos os gráficos são gerados automaticamente em `src/graphics/`:

1. **01_feature_importance.png** - Importância das 11 features
2. **02_metricas_desempenho.png** - R², MAE, RMSE, Resumo
3. **03_acuracia_por_faixa.png** - Performance por faixa de ETc
4. **04_distribuicao_erros.png** - Histograma e box plot de erros
5. **05_predicoes_vs_reais.png** - Scatter plot predições vs reais
6. **06_sensibilidade_features.png** - Impacto de cada variável
7. **07_analise_dados_treinamento.png** - Distribuições do dataset
8. **08_resumo_modelo.png** - Dashboard executivo

---

## 🔬 Métodos Científicos Aplicados

### ✅ Validação Rigorosa

```
├── K-Fold Cross Validation (5 folds)
├── Train-Test Split (80-20)
├── Stratified Sampling
├── Outlier Detection (IQR method)
└── Reproducibilidade (random_state fixo)
```

### ✅ Tratamento de Dados

```
├── Limpeza (remoção de espaços, NaN)
├── Normalização (Z-score)
├── Codificação (Label Encoding para categóricas)
├── Feature Engineering (seleção de 11 features principais)
└── Validação de Ranges
```

### ✅ Modelagem Estatística

```
├── Ensemble Methods (Random Forest)
├── Feature Importance Analysis
├── Error Distribution Analysis
├── Sensitivity Analysis
└── Model Generalization Assessment
```

---

## 🤝 Contribuição

Este é um projeto acadêmico (TCC) para demonstrar aplicação prática de Machine Learning e Deep Learning em problemas reais de agricultura.

---

## 📝 Licença

MIT License - Veja LICENSE para detalhes

---

## 👨‍💻 Autor

**Seu Nome**
- Email: seu.email@example.com
- GitHub: [@seuprofile](https://github.com)

---

## 🙏 Agradecimentos

- **Datasets**: Plant Village Dataset, Indian Irrigation Dataset
- **Frameworks**: FastAPI, scikit-learn, TensorFlow
- **Metodologia**: FAO Penman-Monteith para cálculos agrofísicos

---

## 📞 Suporte

Para dúvidas ou problemas:
1. Consulte a documentação em `TECNOLOGIAS_E_BIBLIOTECAS.md`
2. Abra uma issue no GitHub
3. Entre em contato pelo email

---

**Desenvolvido com ❤️ para a agricultura moderna** 🌾✨
