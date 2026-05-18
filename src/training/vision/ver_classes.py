import tensorflow as tf
dataset = tf.keras.utils.image_dataset_from_directory('C:\\TCC - Code\\AgriculturaIA\\plantvillagedataset')
print("\nLISTA EXATA DO TENSORFLOW:")
print(dataset.class_names)