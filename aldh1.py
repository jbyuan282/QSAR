import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import random

def count_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)

def subsample_file(file_path, output_path, num_samples):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    subsampled_lines = random.sample(lines, num_samples)
    with open(output_path, 'w') as output_file:
        for line in subsampled_lines:
            output_file.write(line)

def smiles_to_image_file(smiles_string, output_file_name, size=(300, 300)):
    molecule = Chem.MolFromSmiles(smiles_string)
    Draw.MolToFile(molecule, output_file_name, size=size)

def read_smi_and_generate_images(smi_file, output_dir, label):
    with open(smi_file, 'r') as file:
        for index, line in enumerate(file):
            smiles = line.strip()
            output_file_name = f'{output_dir}/{label}/{index}.png'
            smiles_to_image_file(smiles, output_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molecule Classifier")
    parser.add_argument("--active_smi_path", type=str, required=True, help="Path to the SMI file containing SMILES strings for active molecules.")
    parser.add_argument("--inactive_smi_path", type=str, required=True, help="Path to the SMI file containing SMILES strings for inactive molecules.")
    parser.add_argument("--output_dir", type=str, default="./molecule_images", help="Directory to store generated molecule images.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of convolutional layers in the model.")
    parser.add_argument("--first_filters", type=int, default=32, help="Number of filters in the first convolutional layer.")
    parser.add_argument("--filter_multiplier", type=int, default=2, help="Factor to multiply the number of filters in each subsequent convolutional layer.")
    parser.add_argument("--dense_neurons", type=int, default=64, help="Number of neurons in the dense layer.")

    args = parser.parse_args()

    model_name = f"model_layers_{args.num_layers}_filters_{args.first_filters}_mult_{args.filter_multiplier}_dense_{args.dense_neurons}"
    num_active = count_lines_in_file(args.active_smi_path)
    num_inactive = count_lines_in_file(args.inactive_smi_path)

    # Determine the number of samples to subsample to (min of the two classes)
    num_samples_to_subsample = min(num_active, num_inactive)

    # Subsample the larger dataset if necessary
    if num_active > num_samples_to_subsample:
        subsample_file(args.active_smi_path, "active_subsampled.smi", num_samples_to_subsample)
        active_smi_path = "active_subsampled.smi"
    else:
        active_smi_path = args.active_smi_path

    if num_inactive > num_samples_to_subsample:
        subsample_file(args.inactive_smi_path, "inactive_subsampled.smi", num_samples_to_subsample)
        inactive_smi_path = "inactive_subsampled.smi"
    else:
        inactive_smi_path = args.inactive_smi_path
    
    output_dir = args.output_dir

    os.makedirs(f'{output_dir}/active', exist_ok=True)
    os.makedirs(f'{output_dir}/inactive', exist_ok=True)

    # Generate images from SMILES strings
    read_smi_and_generate_images(active_smi_path, output_dir, 'active')
    read_smi_and_generate_images(inactive_smi_path, output_dir, 'inactive')

    train_dir = output_dir
    test_dir = output_dir

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Adding validation split

train_generator = train_datagen.flow_from_directory(
    output_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Specify as training data
)

validation_generator = train_datagen.flow_from_directory(
    output_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Specify as validation data
)

def create_model(input_shape=(300, 300, 3), num_classes=1, num_layers=4, first_filters=32, filter_multiplier=2, dense_neurons=64):
    model = Sequential()
    filters = first_filters
    model.add(Conv2D(filters, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    for _ in range(1, num_layers):
        filters *= filter_multiplier
        model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(dense_neurons, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = create_model(
    num_layers=args.num_layers,
    first_filters=args.first_filters,
    filter_multiplier=args.filter_multiplier,
    dense_neurons=args.dense_neurons
)

# Model training
checkpoint_path = f"./{model_name}/best_model.h5"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Model evaluation
model.load_weights(checkpoint_path)
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Confusion Matrix
predictions = model.predict(validation_generator)
binary_predictions = (predictions > 0.5).astype(int).flatten()
true_labels = validation_generator.classes

cm = confusion_matrix(true_labels, binary_predictions[:len(true_labels)])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
