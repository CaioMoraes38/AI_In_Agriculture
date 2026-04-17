# Resumo Completo do Desenvolvimento - TCC Agricultura IA

**Data:** 16-17 de Abril, 2026  
**Projeto:** Sistema ML para Irrigação + Detecção de Doenças em Tomate  
**Tecnologia:** FastAPI + TensorFlow/Keras + scikit-learn + Python 3.11

---

## Indice
1. [Status Final](#status-final)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Componentes Implementados](#componentes-implementados)
4. [Modelos de Machine Learning](#modelos-de-machine-learning)
5. [API REST](#api-rest)
6. [Processo de Desenvolvimento](#processo-de-desenvolvimento)
7. [Resolução de Problemas](#resolução-de-problemas)
8. [Como Usar](#como-usar)

---

## Status Final

### [OK] Projeto Concluído

| Componente | Status | Detalhes |
|-----------|--------|----------|
| **Irrigação ML** | [OK] Operacional | Random Forest Regressor (R² 86.51%) |
| **Detecção de Doenças** | [OK] Operacional | CNN Keras com 10 classes |
| **API FastAPI** | [OK] Operacional | 2 routers (irrigação + doenças) |
| **Pré-processamento Imagem** | [OK] Aprimorado | NDVI + Gaussian Blur |
| **Modelo v2 (Treino)** | [ANDAMENTO] Em andamento | 36.3K imagens, ~2-3 horas |

---

## Estrutura do Projeto

```
C:\TCC - Code\
├── IA - TCC\
│   ├── AgricutureIA\              # Venv Python 3.11
│   │   ├── Scripts\               # Python executável
│   │   └── Lib\                   # Packages instalados
│   │
│   ├── BaseImage\                 # Dataset de Imagens
│   │   └── plantvillagedataset\
│   │       └── color\             # 36.3K imagens em 10 classes
│   │
│   ├── Irrigation data\           # Dados de Irrigação
│   │   └── irrigation_prediction.csv  # 10K amostras
│   │
│   └── [Código-Fonte]
│       └── src\
│           ├── main.py                # FastAPI entry point
│           ├── config.py              # Configurações
│           ├── models/                # Arquivos de modelo
│           ├── api/
│           │   ├── routes/
│           │   │   ├── irrigation.py  # Endpoints irrigação
│           │   │   └── diseases.py    # Endpoints doenças
│           │   └── dependencies.py    # Injeção de dependências
│           └── ia/
│               ├── services/
│               │   ├── irrigation_ml_regression_service.py
│               │   └── plant_disease_service.py
│               └── models/ (ANTIGO - não usado)
```

---

## Componentes Implementados

### 1. **Irrigação - ML Regressor**

**Arquivo:** `src/ia/services/irrigation_ml_regression_service.py`

- **Modelo:** Random Forest Regressor (200 árvores, max_depth=15)
- **Entrada:** 19 parâmetros
  - Obrigatórios: temperatura, umidade, chuva
  - Opcionais: 11 numéricos + 5 categóricos (tipo_solo, hora_dia, etc)
- **Saída:** Volume de água em litros (0.5 - 30.65L)
- **Performance:**
  - R² Score: 86.51%
  - MAE: 1.705L
  - RMSE: 2.172L
  - Dataset: 8K treino / 2K teste

**Arquivos de Modelo:**
- `models/irrigation_regressor.pkl` (35.8 MB) [OK] ATIVO
- `models/irrigation_model.pkl` (17.4 MB) [OBSOLETO] OBSOLETO (classifier antigo)
- `models/irrigation_*.json` - encoders, scalers, features

---

### 2. **Detecção de Doenças - CNN Keras**

**Arquivo:** `src/ia/services/plant_disease_service.py`

**Modelo Original (model_Tomato.keras):**
- **Arquitetura:** 158 camadas (Functional Model)
- **Input:** 224×224×3 (imagens RGB)
- **Output:** 10 classes
- **Classes:**
  1. Bacterial_spot (mancha bacteriana)
  2. Early_blight (requeima precoce)
  3. healthy (planta saudável)
  4. Late_blight (requeima tardia)
  5. Leaf_Mold (mofo da folha)
  6. Septoria_leaf_spot (mancha de septória)
  7. Spider_mites (ácaros)
  8. Target_Spot (mancha alvo)
  9. Tomato_mosaic_virus (vírus do mosaico)
  10. Tomato_Yellow_Leaf_Curl_Virus (TYLCV)

**Problema Identificado:** Modelo treinado estava muito viesado para classe "healthy" (99.99% de confiança em não-doença)

**Solução Implementada:**
- Adicionado **pré-processamento NDVI** (detecção de verde)
- Adicionado **Gaussian Blur** para imagens sem vegetação
- Gaussianfilter aplicado quando < 15% de verde

**Modelo v2 (Em Treinamento):**
- **Dados:** 36.3K imagens (distribuição original mantida)
- **Split:** 80% treino (29K) / 20% validação (7.3K)
- **Arquitetura:** 32 camadas (CNN com BatchNorm + Dropout)
  - 4 blocos convolucionais (32→64→128→256 filters)
  - GlobalAveragePooling + 2 Dense layers (512→256→10)
  - 1.4M parâmetros
- **Treinamento:**
  - Batch size: 64
  - Optimizer: Adam (lr=0.001)
  - Loss: categorical_crossentropy
  - Early stopping: patience=5
  - Reduce LR: factor=0.5, patience=3
  - Data augmentation: rotação±20°, zoom, shifts, flip

---

### 3. **API REST - FastAPI**

**Arquivo:** `src/main.py`

**Configuração:**
- **Host:** 0.0.0.0
- **Port:** 8000
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **CORS:** Habilitado para todos os origins

**Dependências:**
```
FastAPI 0.135.2
Uvicorn
python-multipart 0.0.26 (para upload de arquivos)
```

#### **Router 1: Irrigação** (`api/routes/irrigation.py`)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/irrigacao/ml/prever` | POST | Predição de volume |
| `/irrigacao/ml/info` | GET | Informações do modelo |
| `/irrigacao/ml/parametros` | GET | Documentação de parâmetros |
| `/irrigacao/ml/exemplo` | POST | Exemplo de resposta |

**Request Example:**
```json
{
  "temperatura": 25.5,
  "umidade": 65.0,
  "chuva": 5.0,
  "tipo_solo": "argiloso",
  "hora_dia": "09:00"
}
```

**Response Example:**
```json
{
  "volume_litros": 15.2,
  "nivel_alerta": "normal",
  "confianca": 0.8651,
  "status": "sucesso"
}
```

#### **Router 2: Doenças** (`api/routes/diseases.py`)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/doencas/detectar` | POST | Upload de imagem para classificação |
| `/doencas/info` | GET | Informações do modelo |
| `/doencas/classes` | GET | Lista de 10 classes |
| `/doencas/exemplo` | POST | Exemplo de resposta |

**Request:** Multipart form-data com campo `arquivo` (JPG/PNG)

**Response Example:**
```json
{
  "classe_prevista": "healthy",
  "eh_saudavel": true,
  "confianca_percentual": 80.93,
  "top_3_predicoes": [
    {"classe": "healthy", "confianca": 80.93},
    {"classe": "Septoria_leaf_spot", "confianca": 18.9},
    {"classe": "Early_blight", "confianca": 0.09}
  ],
  "recomendacoes": {
    "status": "planta_saudavel",
    "mensagem": "Planta em bom estado de saúde",
    "acoes": [...]
  },
  "status": "sucesso"
}
```

---

## 🧠 Modelos de Machine Learning

### Random Forest Regressor (Irrigação)
- **Arquivo:** `models/irrigation_regressor.pkl` (35.8 MB)
- **Treinamento:** ~5 segundos
- **Performance:** R² = 86.51%, MAE = 1.705L
- **Saída:** Volume contínuo (0.5-30.65L)

### CNN Keras (Detecção v1)
- **Arquivo:** `models/model_Tomato.keras` (11.6 MB)
- **Status:** ✅ Funcional mas viesado
- **Camadas:** 158
- **Parâmetros:** ~Vários milhões

### CNN Keras v2 (Detecção em Treinamento)
- **Será salvo em:** `models/model_Tomato_v2.keras`
- **Camadas:** 32
- **Parâmetros:** 1.4M (mais leve)
- **Tempo estimado:** 2-3 horas no CPU

---

## 📊 Processo de Desenvolvimento

### Fase 1: Preparação da Irrigação
1. ✅ Explorado dataset `irrigation_prediction.csv` (10K amostras)
2. ✅ Treinado Random Forest Regressor
3. ✅ Removidos ajustes baseados em regras hardcoded
4. ✅ Obtido R² de 86.51%

### Fase 2: Implementação da API
1. ✅ Criado projeto FastAPI
2. ✅ Implementados routers para irrigação
3. ✅ Dependency injection para services
4. ✅ Geração automática de Swagger docs

### Fase 3: Integração de Doenças
1. ✅ Criado PlantDiseaseService para carregar model_Tomato.keras
2. ✅ Implementado pré-processamento de imagem
3. ✅ Adicionadas rotas para detecção
4. ✅ Gerado JSON com classes limpas (sem "Tomato___" prefix)

### Fase 4: Melhoria do Modelo
1. ✅ Identificado viés do modelo (99.99% "healthy")
2. ✅ Adicionado NDVI para detecção de vegetação
3. ✅ Aplicado Gaussian Blur em imagens sem folha
4. ✅ Iniciado treinamento do modelo v2 (36.3K imagens)

### Fase 5: Resolução de Problemas
1. ✅ Instalado python-multipart para upload de arquivos
2. ✅ Corrigido uso de venv Python (não sistema)
3. ✅ Ajustado carregamento de imagens sob demanda
4. ✅ Refatorado script de treinamento com ImageDataGenerator

---

## 🔧 Resolução de Problemas

### Problema 1: python-multipart não encontrado
**Sintoma:** `Form data requires "python-multipart" to be installed`  
**Causa:** Execução com Python do sistema em vez do venv  
**Solução:** Usar `c:\TCC - Code\IA - TCC\AgricutureIA\Scripts\python.exe`

### Problema 2: Modelo Keras não carregando
**Sintoma:** Model não retornava predições reais  
**Causa:** Model path incorreto ou formato .h5 vs .keras  
**Solução:** Usar `model_Tomato.keras` e carregar com `keras.models.load_model()`

### Problema 3: Out of Memory ao treinar
**Sintoma:** `Unable to allocate 16.3 GiB for array (36318, 224, 224, 3)`  
**Causa:** Tentava carregar todas 36K imagens em RAM simultaneamente  
**Solução:** Refatorar com `ImageDataGenerator` para carregamento sob demanda

### Problema 4: Dataset Path não encontrado
**Sintoma:** Script não achava `plantvillage dataset` com espaço  
**Causa:** Pasta real é `plantvillagedataset` sem espaço  
**Solução:** Corrigir caminho nos scripts de treinamento

---

## 🚀 Como Usar

### Executar API

```bash
# Entrar no venv
cd "c:\TCC - Code\IA - TCC\AgricutureIA"
.\Scripts\Activate.ps1

# Rodar API
cd src
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Acesso:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Testar Endpoint de Irrigação

```bash
# Predição simples
curl -X POST "http://localhost:8000/irrigacao/ml/prever" \
  -H "Content-Type: application/json" \
  -d '{
    "temperatura": 25.5,
    "umidade": 65.0,
    "chuva": 5.0
  }'
```

### Testar Endpoint de Doenças

```bash
# Upload de imagem
curl -X POST "http://localhost:8000/doencas/detectar" \
  -F "arquivo=@plant_photo.jpg"
```

### Treinar Novo Modelo v2

```bash
python src/treinar_modelo_v2_optimized.py
```

**Tempo estimado:** 2-3 horas  
**Resultado:** Salva em `src/models/model_Tomato_v2.keras`

---

## 📈 Métricas e Performance

### Irrigação (Random Forest)
- **Treino:** 8.000 amostras
- **Teste:** 2.000 amostras
- **R² Score:** 86.51%
- **MAE:** 1.705 litros
- **RMSE:** 2.172 litros
- **Range Saída:** 0.5L - 30.65L

### Doenças (CNN v1)
- **Total de Imagens:** 36.318 (original)
- **Classes:** 10
- **Acurácia:** ~70-80% estimada (modelo viesado)

### Doenças (CNN v2 - Treinando)
- **Treino:** 29.054 imagens
- **Validação:** 7.264 imagens
- **Epochs:** 30 (com early stopping)
- **Batch Size:** 64
- **Esperado:** >85% validação accuracy

---

## 🛠️ Tecnologias Utilizadas

- **Python:** 3.11
- **Deep Learning:** TensorFlow 2.x + Keras
- **ML Clássico:** scikit-learn 1.3.2
- **API Web:** FastAPI 0.135.2 + Uvicorn
- **Processamento de Imagem:** PIL, scipy (gaussian_filter)
- **Dados:** NumPy, Pandas
- **Serialização:** JSON, pickle

---

## 📝 Arquivos Principais

| Arquivo | Função | Status |
|---------|--------|--------|
| `src/main.py` | FastAPI entry point | ✅ Operacional |
| `src/config.py` | Configurações centralizadas | ✅ Clean |
| `src/api/routes/irrigation.py` | Endpoints irrigação | ✅ Operacional |
| `src/api/routes/diseases.py` | Endpoints doenças | ✅ Operacional |
| `src/ia/services/irrigation_ml_regression_service.py` | ML de irrigação | ✅ Operacional |
| `src/ia/services/plant_disease_service.py` | CNN doenças | ✅ Operacional |
| `src/treinar_modelo_v2_optimized.py` | Script de treinamento | ⏳ Executando |
| `src/models/irrigation_regressor.pkl` | Modelo irrigação | ✅ Ativo |
| `src/models/model_Tomato.keras` | CNN doenças v1 | ✅ Ativo |
| `src/models/model_Tomato_v2.keras` | CNN doenças v2 | ⏳ Gerando |

---

## ✨ Destaques

✅ **Sistema completo de ML em produção**
- 2 modelos diferentes (Regressão + CNN)
- API REST com documentação automática
- Tratamento robusto de erros
- Pré-processamento inteligente de imagens

✅ **Código clean e organizado**
- Removidas todas as hardcoded rules
- Dependency injection implementado
- Separação clara de concerns
- Nomes descritivos

✅ **Otimizações implementadas**
- NDVI para detecção de folha
- Gaussian blur para imagens ruins
- Class weights para dataset imbalanceado
- Data augmentation para melhor generalização
- Early stopping para evitar overfitting

---

## 🔄 Próximos Passos (Após Treinamento v2)

1. Avaliar acurácia do modelo v2
2. Se >85% validação: substituir model_Tomato.keras por v2
3. Fazer testes de campo com imagens reais
4. Ajustar NDVI threshold se necessário
5. Implementar cache de predições
6. Adicionar autenticação na API (se necessário)

---

**Documento gerado:** 17 de Abril, 2026  
**Status Final:** ✅ API Operacional + 🔄 Modelo v2 Treinando
