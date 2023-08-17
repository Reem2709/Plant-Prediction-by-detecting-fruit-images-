# Import necessary libraries
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Set the path to the data directory
data_dir = 'data/fruits/'

# Get the list of fruit classes
fruit_classes = os.listdir(data_dir)

# Set the parameters for the CNN model
img_width, img_height = 150, 150
batch_size = 32
epochs = 10

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(fruit_classes), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up the data generator for training
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

# Train the model
history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples//batch_size,
        epochs=epochs)

# Save the trained model
model.save('data/generated/model.h5')

# Use the model to make predictions on a new image
# Load the saved model
loaded_model = keras.models.load_model('data/generated/model.h5')

# Load the new image to classify
new_image_path = 'data/test/7.jpg'
new_image = tf.keras.utils.load_img(new_image_path, target_size=(img_width, img_height))
new_image_array = tf.keras.utils.img_to_array(new_image)
new_image_array = tf.expand_dims(new_image_array, 0) # Add batch dimension
predicted_class = loaded_model.predict(new_image_array)

# Get the predicted class label
predicted_class_index = predicted_class.argmax()
if predicted_class_index >= len(fruit_classes):
    print("No match found")
else:
    predicted_class_label = fruit_classes[predicted_class_index]
    print(f"The plant in the image is: {predicted_class_label}"+" plant")
