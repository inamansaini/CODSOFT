import numpy as np
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Verify the directory contents
screenshots_dir = r'C:\Users\naman\OneDrive\Pictures\Screenshots'
files = os.listdir(screenshots_dir)
print("Files in directory:", files)

# Load and preprocess data
def load_data():
  image_paths = [
      r'C:\Users\naman\OneDrive\Pictures\Screenshots\image1.jpg',
      r'C:\Users\naman\OneDrive\Pictures\Screenshots\image2.jpg'
  ]  # Actual paths to your images
  captions = ['Caption for image 1', 'Caption for image 2']  # Corresponding captions
  return image_paths, captions

def preprocess_captions(captions):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(captions)
  vocab_size = len(tokenizer.word_index) + 1
  sequences = tokenizer.texts_to_sequences(captions)
  max_sequence_length = max(len(seq) for seq in sequences)
  padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
  return tokenizer, vocab_size, max_sequence_length, padded_sequences

# Feature extraction using ResNet50
def extract_features(img_path, model):
  img = image.load_img(img_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = preprocess_input(img_array)
  features = model.predict(img_array)
  return features.flatten()  # Flatten to 1D array

# Reshape features (**Fix applied here**):
def reshape_features(features, timesteps=1):
  return features.reshape((features.shape[0], timesteps, features.shape[1]))

# Create and compile the caption generation model
def create_captioning_model(vocab_size, embedding_dim, max_sequence_length, feature_dim):
  model = Sequential()
  model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
  model.add(LSTM(256, return_sequences=True))
  model.add(Dropout(0.5))
  model.add(LSTM(256))
  model.add(Dense(vocab_size, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  return model

# Generate captions based on features
def generate_caption(features, model, tokenizer, max_sequence_length):
  # Placeholder function to demonstrate caption generation
  # Replace with actual caption generation code
  return "Generated caption"

# Main function to train and test the model
def main():
  # Load data
  image_paths, captions = load_data()

  # Preprocess captions
  tokenizer, vocab_size, max_sequence_length, padded_sequences = preprocess_captions(captions)

  # Extract features from images
  resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
  feature_dim = resnet_model.output_shape[1]
  features = np.array([extract_features(img_path, resnet_model) for img_path in image_paths])

  # Print the shape of features before reshaping
  print("Shape of features before reshaping:", features.shape)

  # Reshape features for LSTM (**Using timesteps=1**)
  features = reshape_features(features, timesteps=1)

  # Print the shape of features after reshaping
  print("Shape of features after reshaping:", features.shape)

  # Split data into training and testing sets
  X_train, X
