#1. Setup and Import Libraries
#First, we'll import the necessary libraries for data handling, audio processing, and machine learning.

# Import libraries
import pandas as pd
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#2. Load and Explore the Data
#We load the train and test datasets to understand the structure of the data.

# Paths to data
train_csv_path = r"c:\Users\JOAQUIM CELESTINO\Downloads\DataSCIENCEPACK\AUDIO DATA\train_fuSp8nd.csv"
test_csv_path = r"c:\Users\JOAQUIM CELESTINO\Downloads\DataSCIENCEPACK\AUDIO DATA\test_B0QdNpj.csv"
audio_path = r"c:\Users\JOAQUIM CELESTINO\Downloads\DataSCIENCEPACK\AUDIO DATA\Train"
audio_path1 = r"C:\Users\JOAQUIM CELESTINO\Downloads\DataSCIENCEPACK\AUDIO DATA\Test"

# Load Train and Test Data
train_data = pd.read_csv(train_csv_path)
test_data = pd.read_csv(test_csv_path)

# Check the structure
print(train_data)
print(test_data)

#3. Preprocess the Data
#Extract features using Librosa
#We'll extract meaningful audio features such as MFCC (Mel-frequency cepstral coefficients), which are widely used in audio classification tasks.

# Function to extract features from an audio file
def extract_features(audio_path):
    try:
        audio, sample_rate = librosa.load(audio_path, duration=4.0)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {audio_path}, Error: {str(e)}")
        return None

#Apply feature extraction to the dataset

# Extract features and labels from the training data
X_train = []
y_train = []

print("Extracting features from training audio files...")
for i, row in train_data.iterrows():
    # Ensure the ID is converted to a string before appending '.wav'
    file_path = os.path.join(audio_path, str(row['ID']) + '.wav')  # Assuming audio files have .wav extension
    features = extract_features(file_path)
    if features is not None:
        X_train.append(features)
        y_train.append(row['Class'])

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

#Encode the labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)

# Train-test split for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_split, y_train_split)

# Validate the model
y_val_pred = clf.predict(X_val_split)
print("Validation Accuracy:", accuracy_score(y_val_split, y_val_pred))
print("Classification Report:\n", classification_report(y_val_split, y_val_pred, target_names=encoder.classes_))

X_test = []
print("Extracting features from test audio files...")
for i, row in test_data.iterrows():
    # Ensure the ID is converted to a string before appending '.wav'
    file_path = os.path.join(audio_path1, str(row['ID']) + '.wav')  # Adjust path if needed
    features = extract_features(file_path)
    if features is not None:
        X_test.append(features)

# Convert to numpy array and standardize
X_test = np.array(X_test)
X_test = scaler.transform(X_test)

# Predict on the test set
y_test_pred = clf.predict(X_test)
y_test_pred = encoder.inverse_transform(y_test_pred)

# Create a submission file
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'Class': y_test_pred
})

submission_path = r"c:\Users\JOAQUIM CELESTINO\Downloads\DataSCIENCEPACK\AUDIO DATA\submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission saved to:", submission_path)
