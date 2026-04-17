"""
Script de treinamento de novo modelo CNN para detecção de doenças em tomate (v2)
Usa ImageDataGenerator para carregar imagens sob demanda (economia de memória)
"""
import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from datetime import datetime
import shutil

# Configurações
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 30

# Encontrar caminho do dataset
current_dir = Path(__file__).parent  # src/
agriculture_dir = current_dir.parent  # AgricutureIA/
project_root = agriculture_dir.parent  # IA - TCC/
DATASET_PATH = project_root / "BaseImage" / "plantvillagedataset" / "color"
MODELS_DIR = current_dir / "models"
TEMP_DIR = current_dir / "temp_dataset"

print("="*60)
print("TREINAMENTO DE NOVO MODELO CNN v2 (Com ImageDataGenerator)")
print("="*60)

# Verificar se dataset existe
if not DATASET_PATH.exists():
    print(f"[ERRO] Dataset nao encontrado: {DATASET_PATH}")
    exit(1)

print(f"[OK] Dataset encontrado: {DATASET_PATH.absolute()}")

# Listar classes
classes_dirs = sorted([d for d in DATASET_PATH.iterdir() if d.is_dir()])
classes = [d.name.replace('Tomato___', '') for d in classes_dirs]

print(f"[OK] Classes encontradas: {len(classes)}")
total_images = 0
for cls_dir, cls_clean in zip(classes_dirs, classes):
    img_count = len(list(cls_dir.glob("*.jpg"))) + len(list(cls_dir.glob("*.JPG")))
    total_images += img_count
    print(f"  - {cls_clean:40s} {img_count:5d} imagens")

print(f"\n[OK] Total de imagens: {total_images}")

# Preparar estrutura de diretórios para ImageDataGenerator
print("\nPreparando estrutura de dados...")

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)

TEMP_DIR.mkdir(exist_ok=True)
train_dir = TEMP_DIR / "train"
val_dir = TEMP_DIR / "val"

train_dir.mkdir(exist_ok=True)
val_dir.mkdir(exist_ok=True)

# Criar subpastas para cada classe e distribuir imagens
for cls_dir, cls_clean in zip(classes_dirs, classes):
    (train_dir / cls_clean).mkdir(exist_ok=True)
    (val_dir / cls_clean).mkdir(exist_ok=True)

# Distribuir imagens (80% treino, 20% validação)
print("Distribuindo imagens entre treino e validação...")
images_processed = 0

for cls_dir, cls_clean in zip(classes_dirs, classes):
    imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG"))
    np.random.shuffle(imgs)
    
    split_idx = int(len(imgs) * 0.8)
    
    for img_path in imgs[:split_idx]:
        try:
            shutil.copy(img_path, train_dir / cls_clean / img_path.name)
            images_processed += 1
            if images_processed % 1000 == 0:
                print(f"  Processadas {images_processed} imagens...")
        except Exception as e:
            print(f"  [AVISO] Erro ao copiar {img_path}: {e}")
    
    for img_path in imgs[split_idx:]:
        try:
            shutil.copy(img_path, val_dir / cls_clean / img_path.name)
            images_processed += 1
            if images_processed % 1000 == 0:
                print(f"  Processadas {images_processed} imagens...")
        except Exception as e:
            print(f"  [AVISO] Erro ao copiar {img_path}: {e}")

print(f"\n[OK] Distribuição concluída: {images_processed} imagens")

# Criar ImageDataGenerators com augmentação
print("\nCriando data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Carregar dados via generators
train_generator = train_datagen.flow_from_directory(
    str(train_dir),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    str(val_dir),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"[OK] Train generator: {train_generator.samples} imagens")
print(f"[OK] Val generator: {val_generator.samples} imagens")

# Calcular class weights
class_counts = {}
for cls_name in classes:
    train_count = len(list((train_dir / cls_name).glob("*.jpg"))) + len(list((train_dir / cls_name).glob("*.JPG")))
    class_counts[cls_name] = train_count

total = sum(class_counts.values())
class_weights = {i: total / (len(classes) * count) for i, count in enumerate(class_counts.values())}

print(f"\n[OK] Class weights (para balanceamento):")
for i, weight in class_weights.items():
    print(f"  Classe {i}: {weight:.3f}")

# Criar modelo CNN melhorado
print("\n[OK] Criando arquitetura CNN v2...")

model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=IMG_SIZE + (3,)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 4
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Global Average Pooling
    layers.GlobalAveragePooling2D(),
    
    # Dense layers
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    # Output
    layers.Dense(len(classes), activation='softmax')
])

# Compilar modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Arquitetura criada: {len(model.layers)} camadas")
print(f"  Parâmetros totais: {model.count_params():,}")

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Treinar
print("\n" + "="*60)
print("INICIANDO TREINAMENTO")
print("="*60)

steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = val_generator.samples // BATCH_SIZE

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Avaliar
print("\n" + "="*60)
print("AVALIAÇÃO FINAL")
print("="*60)

train_loss, train_acc = model.evaluate(train_generator, steps=steps_per_epoch, verbose=0)
val_loss, val_acc = model.evaluate(val_generator, steps=validation_steps, verbose=0)

print(f"Treino - Loss: {train_loss:.4f}, Acurácia: {train_acc:.4f}")
print(f"Validação - Loss: {val_loss:.4f}, Acurácia: {val_acc:.4f}")

# Salvar modelo v2
output_path = MODELS_DIR / "model_Tomato_v2.keras"
model.save(str(output_path))
print(f"\n[OK] Modelo salvo: {output_path}")
print(f"  Tamanho: {output_path.stat().st_size / (1024*1024):.2f} MB")

# Salvar informações de treinamento
info = {
    'version': 'v2',
    'timestamp': datetime.now().isoformat(),
    'classes': classes,
    'total_images': total_images,
    'train_images': train_generator.samples,
    'val_images': val_generator.samples,
    'epochs_trained': len(history.history['loss']),
    'final_train_accuracy': float(train_acc),
    'final_val_accuracy': float(val_acc),
    'final_train_loss': float(train_loss),
    'final_val_loss': float(val_loss),
    'img_size': IMG_SIZE,
    'batch_size': BATCH_SIZE
}

info_path = MODELS_DIR / "model_Tomato_v2_info.json"
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)

print(f"[OK] Informações salvas: {info_path}")

# Limpar diretório temporário
print("\nLimpando diretório temporário...")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)

print("\n" + "="*60)
print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("="*60)
