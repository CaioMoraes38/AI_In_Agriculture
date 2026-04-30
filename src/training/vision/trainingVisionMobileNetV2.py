import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 10 
EPOCHS = 10


CAMINHO_DATASET = "plantvillagedataset/color" 



PASTA_DESTINO = "src/models/modelsV1"
NOME_ARQUIVO = "modelo_mobilenet_tomate.keras" 
CAMINHO_COMPLETO = os.path.join(PASTA_DESTINO, NOME_ARQUIVO)

os.makedirs(PASTA_DESTINO, exist_ok=True)


print("Carregando dataset de treinamento...")
dataset_treino = image_dataset_from_directory(
    CAMINHO_DATASET,
    validation_split=0.2, 
    subset="training",   
    seed=123,             
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("\nCarregando dataset de validação...")
dataset_validacao = image_dataset_from_directory(
    CAMINHO_DATASET,
    validation_split=0.2,
    subset="validation", 
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE
dataset_treino = dataset_treino.prefetch(buffer_size=AUTOTUNE)
dataset_validacao = dataset_validacao.prefetch(buffer_size=AUTOTUNE)


modelo_base = MobileNetV2(
    input_shape=IMG_SIZE + (3,), 
    include_top=False, 
    weights='imagenet' 
)

modelo_base.trainable = False 

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = modelo_base(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

modelo = Model(inputs, outputs)


modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("\nIniciando o treinamento...")
historico = modelo.fit(
    dataset_treino,
    validation_data=dataset_validacao, # Avalia o modelo a cada época
    epochs=EPOCHS
)


modelo.save(CAMINHO_COMPLETO)
print(f"\n✅ Treinamento concluído! O modelo foi salvo com sucesso em: {CAMINHO_COMPLETO}")