import os
from PIL import Image
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import joblib

# Read Folder
folder_path = '..\Sistem Pengenalan Digit\DataSet_03'

# HOG Feature Extraction
def extract_hog_features(image_np, target_size=(128, 128)):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize(target_size)  # Resize the image
    image_np = np.array(image) / 255.0  # Rescale pixel values to [0, 1]
    features, hog_image = hog(image_np, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

hog_features = []
labels = []

file_count = 0
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".jpg"):
            file_count += 1
            image_path = os.path.join(root, file)
            features = extract_hog_features(image_path)
            if features is not None:
                label = int(file[0])
                hog_features.append(features)
                labels.append(label)


hog_features = np.array(hog_features)
labels = np.array(labels)

print(f"Total images: {len(hog_features)}")

def evaluate_model(train_size):
    X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, train_size=train_size, random_state=42)
    
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return svm_model, accuracy

train_sizes = [0.6, 0.7, 0.8]
models = {}
accuracies = {}

for train_size in train_sizes:
    model, accuracy = evaluate_model(train_size)
    models[train_size] = model
    accuracies[train_size] = accuracy

for train_size in train_sizes:
    model_path = f'SVM_{int(train_size*100)}.pkl'
    joblib.dump(models[train_size], model_path)
print("Save Model Selesai")
