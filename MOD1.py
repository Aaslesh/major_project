import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize
import shutil

# Loading Model
model = tf.keras.models.load_model("Trained_model.h5")
# Classes
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds

    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    # Iterate over each chunk
    for i in range(num_chunks):
        # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples

        # Extract the chunk of audio
        chunk = audio_data[start:end]

        # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)

        # Resize the Mel spectrogram
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

# Model Prediction
def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# Process all files in a folder
def process_folder(folder_path):
    classified_folder = os.path.join(folder_path, 'classified')

    # Create classified folder if it doesn't exist
    if not os.path.exists(classified_folder):
        os.makedirs(classified_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)

            # Preprocess the audio file
            X_test = load_and_preprocess_data(file_path)

            # Predict the class
            c_index = model_prediction(X_test)
            predicted_class = classes[c_index]
            print(f"Model Prediction for {filename} :: Music Genre --> {predicted_class}")

            # Create the class folder inside the classified folder
            dest_folder = os.path.join(classified_folder, predicted_class)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            # Move the file to the predicted class folder
            dest_path = os.path.join(dest_folder, filename)
            shutil.move(file_path, dest_path)
            print(f"Moved {filename} to {dest_path}")

# Path to the folder containing audio files
folder_path = r"A:\study\PROJECTS\MAJOUR"

# Set the environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Process the folder
process_folder(folder_path)
