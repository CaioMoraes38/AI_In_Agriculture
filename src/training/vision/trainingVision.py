import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  
NUM_CLASSES = 10 
DATASET_PATH = "plantvillagedataset/color"
SAVE_PATH = "src/models/modelsVisionV2"
MODEL_NAME = "modelo_custom_tomate.keras"
FULL_SAVE_PATH = os.path.join(SAVE_PATH, MODEL_NAME)

os.makedirs(SAVE_PATH, exist_ok=True)


print("--- Lendo imagens do dataset ---")
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


print("--- Construindo a arquitetura ---")
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5), # Evita que o modelo "decore" as fotos
    layers.Dense(NUM_CLASSES, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary() # Exibe a tabela com a estrutura do modelo

print("\n--- Iniciando Treinamento ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


print(f"\n--- Salvando modelo em: {FULL_SAVE_PATH} ---")
model.save(FULL_SAVE_PATH)
print("✅ Tudo pronto! Modelo salvo e pronto para uso.")