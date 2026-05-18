import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

MODEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'visionModel', 'plant_disease_vision_model.keras')

IMG_SIZE = (224, 224)

CLASSES_ORIGINAIS = [
    'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

TRADUCAO_DOENCAS = {
    "Tomato___Bacterial_spot": "Mancha Bacteriana",
    "Tomato___Early_blight": "Requeima Precoce",
    "Tomato___Late_blight": "Requeima Tardia",
    "Tomato___Leaf_Mold": "Mofo da Folha",
    "Tomato___Septoria_leaf_spot": "Mancha Septória",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Ácaro Rajado",
    "Tomato___Target_Spot": "Mancha Alvo",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Vírus do Enrolamento",
    "Tomato___Tomato_mosaic_virus": "Vírus do Mosaico",
    "Tomato___healthy": "Saudável (Sem doenças)"
}

def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Erro: Modelo não encontrado em {MODEL_PATH}")
        print("Certifique-se de que o treinamento foi concluído e o arquivo .keras existe.")
        return None
    
    print("⏳ Carregando a Inteligência Artificial...")
    return tf.keras.models.load_model(MODEL_PATH)

def prever_imagem(caminho_imagem, modelo):
    caminho_imagem = caminho_imagem.strip('"').strip("'")
    if not os.path.exists(caminho_imagem):
        print(f"❌ Erro: Imagem não encontrada no caminho '{caminho_imagem}'")
        return

    try:
        img = tf.keras.utils.load_img(caminho_imagem, target_size=IMG_SIZE)
        
        img_array = tf.keras.utils.img_to_array(img)
        
        img_batch = np.expand_dims(img_array, axis=0)

        predicoes = modelo.predict(img_batch, verbose=0) 
        
        indice_classe = np.argmax(predicoes[0])
        confianca = predicoes[0][indice_classe] * 100
        
        classe_original = CLASSES_ORIGINAIS[indice_classe]
        diagnostico = TRADUCAO_DOENCAS.get(classe_original, classe_original)

        print("\n" + "-"*40)
        print(" 🍅 RESULTADO DA ANÁLISE")
        print("-"*40)
        print(f"Diagnóstico : {diagnostico}")
        print(f"Confiança   : {confianca:.2f}%")
        print(f"ID da Classe: {classe_original}")
        print("-"*40 + "\n")

    except Exception as e:
        print(f"❌ Erro ao processar a imagem: {e}")

def main():
    print("="*60)
    print(" TESTADOR DO MODELO DE DOENÇAS EM TOMATES")
    print("="*60)
    
    modelo = carregar_modelo()
    if modelo is None:
        return

    print("\n✅ Modelo carregado com sucesso!")
    print("DICA: Você pode arrastar a imagem do Windows direto para este terminal para colar o caminho.")

    while True:
        caminho = input("Caminho da imagem (ou digite 'sair' para fechar): ")
        
        if caminho.lower() == 'sair':
            print("Encerrando testador...")
            break
            
        if caminho.strip() == "":
            continue
            
        prever_imagem(caminho, modelo)

if __name__ == '__main__':
    main()