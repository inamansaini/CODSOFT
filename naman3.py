# Import necessary libraries
import numpy as np
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load and preprocess data
def load_data():
    # Replace with actual paths and data
    image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg']  # Example image paths
    captions = ['Caption for image 1', 'Caption for image 2']  # Example captions
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
    return features

# Create and compile the caption generation model
def create_captioning_model(vocab_size, embedding_dim, max_sequence_length):
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
    features = np.array([extract_features(img_path, resnet_model) for img_path in image_paths])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, padded_sequences, test_size=0.2)
    
    # Create and train the model
    embedding_dim = 100
    captioning_model = create_captioning_model(vocab_size, embedding_dim, max_sequence_length)
    captioning_model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Test the model
    for img_path in image_paths:
        features = extract_features(img_path, resnet_model)
        caption = generate_caption(features, captioning_model, tokenizer, max_sequence_length)
        print(f"Caption for {img_path}: {caption}")

if __name__ == "__main__":
    main()
