from tensorflow.keras import models, layers, optimizers, preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf

data_dir = r"/data/"

# Define some parameters
img_width, img_height = 40, 40
batch_size = 32

# Load the training and validation set
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Define the output classes
class_names = train_ds.class_names

# Define the CNN model
model = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fit the model to the dataset
history = model.fit(train_ds, epochs=15, 
                    validation_data=val_ds)

# Save the model for use in main.py
model.save('/models/eye_state')


# Plot the training and validation accuracy 
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label ='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')