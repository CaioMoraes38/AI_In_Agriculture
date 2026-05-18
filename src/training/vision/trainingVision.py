import os
import json
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

DATASET_PATH = os.path.join(PROJECT_ROOT, 'plantvillagedataset')
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'src', 'models', 'visionModel')

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10

def carregar_dados():
    print(f"Carregando imagens de: {DATASET_PATH}")
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    print(f"Classes encontradas: {len(class_names)}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, class_names

def construir_modelo(num_classes):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True 
    
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    x = base_model(x, training=False) 
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model

def salvar_artefatos(model, history, class_names, test_acc):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_SAVE_DIR, 'plant_disease_vision_model.keras')
    model.save(model_path)
    print(f"\n✅ Modelo salvo em: {model_path}")

    history_path = os.path.join(MODEL_SAVE_DIR, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✅ Histórico salvo em: {history_path}")
    metrics_path = os.path.join(MODEL_SAVE_DIR, 'vision_model_metrics.json')
    metrics_data = {
        "model_type": "MobileNetV2 Transfer Learning",
        "num_classes": len(class_names),
        "image_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "train_accuracy": float(history.history['accuracy'][-1]),
        "test_accuracy": float(test_acc),
        "training_date": datetime.now().isoformat()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"✅ Métricas JSON salvas em: {metrics_path}")

def main():
    print("="*50)
    print(" INICIANDO TREINAMENTO DE VISÃO COMPUTACIONAL")
    print("="*50)

    train_dataset, val_dataset, class_names = carregar_dados()
    
    model = construir_modelo(len(class_names))
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    print("\nIniciando treinamento...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )

    print("\nAvaliando modelo com dados de validação...")
    val_loss, val_acc = model.evaluate(val_dataset)
    print(f"Acurácia final de validação: {val_acc:.4f}")

    salvar_artefatos(model, history, class_names, val_acc)
    
    print("\n🚀 Treinamento concluído! Agora você pode rodar o script de gráficos.")

if __name__ == '__main__':
    main()