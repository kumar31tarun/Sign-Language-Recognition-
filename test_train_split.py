import os
#library for managing files path like copy, move and so on
import shutil
from sklearn.model_selection import train_test_split

# Define directories
augmented_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'augmented_data'))
train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'train_data'))
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data'))

# Create train and test directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Loop through each symbol class in the augmented data
for symbol in os.listdir(augmented_dir):
    symbol_dir = os.path.join(augmented_dir, symbol)
    if os.path.isdir(symbol_dir):
        # Get the list of augmented images for the current symbol
        symbol_images = [os.path.join(symbol_dir, f) for f in os.listdir(symbol_dir) if os.path.isfile(os.path.join(symbol_dir, f))]

        # Split the images into train and test sets the random value as something which if replicated will give the same output good for reproducibility
        #on left side the first variable is asssumed as training data and second as test by the funtion train_test_split
        train_images, test_images = train_test_split(symbol_images, test_size=0.2, random_state=42)

        # Move train images to train directory
        train_symbol_dir = os.path.join(train_dir, symbol)
        if not os.path.exists(train_symbol_dir):
            os.makedirs(train_symbol_dir)

        for image_path in train_images:
            #saves the basename of the file into image_filename
            image_filename = os.path.basename(image_path)
            #move the location from image_path to train_symbol_dir with the name as image_filename
            shutil.move(image_path, os.path.join(train_symbol_dir, image_filename))

        # Move test images to test directory
        test_symbol_dir = os.path.join(test_dir, symbol)
        if not os.path.exists(test_symbol_dir):
            os.makedirs(test_symbol_dir)

        for image_path in test_images:
            image_filename = os.path.basename(image_path)
            shutil.move(image_path, os.path.join(test_symbol_dir, image_filename))

        # Remove the empty symbol directory from the augmented data
        os.rmdir(symbol_dir)

print("Data split into train and test sets.")
