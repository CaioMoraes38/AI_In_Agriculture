from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict
from ia.services.plant_disease_service import PlantDiseaseService
from api.dependencies import Dependencias

router = APIRouter(prefix="/doencas", tags=["Doenças de Plantas"])


@router.post("/detectar")
async def detectar_doenca(arquivo: UploadFile = File(...)) -> Dict:
    try:
        if arquivo.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
            raise HTTPException(
                status_code=400,
                detail="Formato de arquivo inválido. Use JPG ou PNG."
            )
        
        conteudo = await arquivo.read()
        servico = Dependencias.obter_plant_disease_service()
        
        import base64
        imagem_b64 = base64.b64encode(conteudo).decode('utf-8')
        resultado = servico.detectar_doenca(imagem_b64)
        
        return resultado
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar imagem: {str(e)}"
        )


@router.get("/info")
async def obter_info_modelo() -> Dict:
    servico = Dependencias.obter_plant_disease_service()
    return servico.obter_info()


@router.get("/classes")
async def obter_classes() -> Dict:
    servico = Dependencias.obter_plant_disease_service()
    info = servico.obter_info()
    return {
        'classes': info['classes_disponiveis'],
        'total': info['n_classes'],
        'descricao': [
            {
                'classe': classe,
                'tipo': 'Saudável' if 'healthy' in classe.lower() else 'Doença'
            }
            for classe in info['classes_disponiveis']
        ]
    }


@router.post("/exemplo")
async def exemplo_deteccao() -> Dict:
    return {
        'classe_prevista': 'Tomato___Early_blight',
        'eh_saudavel': False,
        'confianca_percentual': 94.23,
        'top_3_predicoes': [
            {'classe': 'Tomato___Early_blight', 'confianca': 94.23},
            {'classe': 'Tomato___Septoria_leaf_spot', 'confianca': 3.45},
            {'classe': 'Tomato___healthy', 'confianca': 1.12}
        ],
        'recomendacoes': {
            'status': 'doenca_detectada',
            'doenca': 'Tomato___Early_blight',
            'causa': 'Infecção fúngica (Alternaria)',
            'severidade': 'Média',
            'acoes': [
                'Remover folhas basais infectadas',
                'Aplicar fungicida preventivo',
                'Melhorar circulação de ar',
                'Evitar molhar a folhagem'
            ]
        },
        'modelo_usado': 'cnn_keras_tomato',
        'tamanho_imagem': '224x224',
        'status': 'sucesso'
    }
