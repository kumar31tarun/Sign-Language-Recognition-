"""
This code is for making a CNN model which takes input in (28,28) as input size and trains on it with convulational layer and pooling layer in between
"""

import os
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping
#Earling stopping checks the val_loss and if it goes up it stops the training


#input_shape means the input_shape in (height, width, channel) in this channel is gray=1
input_shape = (28, 28, 1)

# classes in the dataset which is 34 classes
NUM_CLASSES = 34

# Creating the CNN model
# using Convolutional layer of 2D with 32 filters and activation 'relu' (Rectified Linear Unit)
# using '64' neurons in dense layer of fully connected layer
#softmax takes the array and converts it into a proability by dividing into x/sum(x)
model = keras.Sequential([
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', groups=1),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', groups=1),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES , activation='softmax')
])

# optimizer determines how model will update its parameters(weight) during training
# Categorial_crossentropy calculates the difference between predicted and actual motivating model to have minimum
# Metrics gives additional information about the model during training like accuracy
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Set the path to your train and test data directories
train_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'train_data'))
test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data'))

# Set the batch size and number of epochs for training
batch_size = 128
epochs = 20

# normalizing the image between 0 and 1 to ensure pizels are within training range
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load and preprocess the training data
train_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(28,28),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
    
)

# Load and preprocess the test data
test_data = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(28,28),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
    
)
#patience=3 means it will wait till 3 epochs and if the val_loss keeps going up it stops
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


#fit() trains the model with train_data and validates it test_data with number of epochs
# Train the model 
model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Evaluate the model on the test data which ouput two values test_loss and test_accuracy
test_loss, test_acc = model.evaluate(test_data)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')

# Save the trained model
model_dir=os.path.dirname(os.path.abspath(__file__))
model.save(os.path.join(model_dir,'sign_language_model.h5'))
