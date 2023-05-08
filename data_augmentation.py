import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

import os

# Define data directory and augmentation directory
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
augmented_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'augmented_data'))


# Define list of symbols and number of samples to generate per class
symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#num_samples tells how many times each image will be augmented
num_samples = 10

# Create a directory for the augmented data
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Define data generator with specified augmentation parameters
#rotation range means 10% rotation, width_shift range means 10% change in horizontal, shearing means tilting, fill mode means filling new pixels with nearset pixels
datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

# Loop through each symbol class and generate augmented samples
for symbol in symbols:
    # Create a directory for the current symbol class in the augmented data directory
    symbol_dir = os.path.join(augmented_dir, symbol)
    if not os.path.exists(symbol_dir):
        os.makedirs(symbol_dir)
    
    # Load the images for the current symbol class
    symbol_path = os.path.join(data_dir, symbol)
    #this checks the path is file or not, if yes creates symbol images path for it
    symbol_images = [os.path.join(symbol_path, f) for f in os.listdir(symbol_path) if os.path.isfile(os.path.join(symbol_path, f))]
    
    # Loop through each image and generate augmented samples
    for i, image_path in enumerate(symbol_images):
        # Load the image and expand its dimensions to match the input shape of the model
        img = keras.preprocessing.image.load_img(image_path, target_size=(28, 28))
        x = keras.preprocessing.image.img_to_array(img)
        x = tf.expand_dims(x, axis=0)
        
        # Generate augmented samples
        j = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=symbol_dir, save_prefix=symbol, save_format='jpg'):
            j += 1
            if j >= num_samples:
                break
                
    print(f"Generated {num_samples} samples for {symbol}")
