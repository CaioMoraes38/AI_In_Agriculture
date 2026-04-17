"""
Rotas de análise visual de plantas
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from api.schemas import ResultadoVisao, ListaClasses, ErroResponse
from api.dependencies import Dependencias
from ia.services.vision_service import VisionService

router = APIRouter(prefix="/visao", tags=["Análise Visual"])


def obter_vision_service() -> VisionService:
    """Dependência para injetar o serviço de visão"""
    return Dependencias.obter_vision_service()


@router.post(
    "/analisar-planta",
    response_model=ResultadoVisao,
    responses={500: {"model": ErroResponse}}
)
async def analisar_planta(
    foto: UploadFile = File(...),
    service: VisionService = Depends(obter_vision_service)
):
    """
    Analisa uma imagem de planta e retorna diagnóstico
    
    - **foto**: Arquivo de imagem (JPEG, PNG, etc.)
    
    Returns:
    - **diagnostico_principal**: Diagnóstico principal da IA
    - **confianca_porcentagem**: Nível de confiança da previsão
    - **top_3_previsoes**: Três melhores previsões
    """
    if foto.size is None or foto.size == 0:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio")
    
    if not foto.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: {foto.content_type}"
        )
    
    try:
        conteudo = await foto.read()
        resultado = service.analisar(conteudo)
        return resultado
        
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erro ao processar a imagem: {str(e)}"
        )


@router.get(
    "/classes",
    response_model=ListaClasses
)
async def obter_classes(service: VisionService = Depends(obter_vision_service)):
    """
    Retorna lista de todas as doenças/condições que o modelo pode detectar
    
    Returns:
    - **total_classes**: Número total de classes
    - **classes**: Lista com nomes de todas as classes
    """
    classes = service.obter_classes()
    
    return {
        "total_classes": len(classes),
        "classes": classes,
        "status": "sucesso"
    }


@router.get("/saude")
async def saude_visao(service: VisionService = Depends(obter_vision_service)):
    """
    Verifica se o serviço de visão está funcionando corretamente
    
    Returns:
    - **status**: Estado do serviço
    - **modelo_carregado**: Se o modelo foi carregado
    - **total_classes**: Número de classes disponíveis
    """
    classes = service.obter_classes()
    
    return {
        "servico": "vision",
        "status": "ativo" if classes else "inativo",
        "modelo_carregado": service.modelo_loader.modelo is not None,
        "total_classes": len(classes)
    }
