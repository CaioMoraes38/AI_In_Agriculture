from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import json
import io
from PIL import Image
import os




@app.post("/analisar-planta")
async def analisar_planta(foto: UploadFile = File(...)):
    if modelo_visao is None:
        raise HTTPException(status_code=500, detail="A IA não carregou corretamente no servidor.")

    try:
        conteudo = await foto.read()
        imagem = Image.open(io.BytesIO(conteudo))
        
        if imagem.mode != "RGB":
            imagem = imagem.convert("RGB")
            
        imagem = imagem.resize((224, 224))
        imagem_array = tf.keras.utils.img_to_array(imagem)
        imagem_array = tf.expand_dims(imagem_array, 0)
        
        predicoes = modelo_visao.predict(imagem_array, verbose=0)
        indice_vencedor = np.argmax(predicoes[0])
        confianca = float(np.max(predicoes[0])) * 100
        
        return {
            "status": "sucesso",
            "arquivo_recebido": foto.filename,
            "diagnostico": nomes_classes[indice_vencedor],
            "confianca_porcentagem": round(confianca, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar a imagem: {str(e)}")