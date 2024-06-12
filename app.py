import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import joblib
import cv2
import numpy as np
from skimage.feature import hog

# Function Load Model
def load_model(model_path):
    return joblib.load(model_path)

# Function HOG Feature Extraction


def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (128, 128))
    features = hog(resized_image, orientations=9, pixels_per_cell=(
        8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

# Prediction Images


def predict_image(image_path, model):
    features = extract_hog_features(image_path)
    if features is not None:
        features = np.array([features])
        prediction = model.predict(features)
        return prediction[0]
    return None


def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

        # Predict using model KNN
        predictions = {}
        for train_size, model in model_knn.items():
            predictions[train_size] = predict_image(file_path, model)

        # Display predictions KNN
        result_text_knn = "Predicted KNN:\n"
        for train_size, prediction in predictions.items():
            result_text_knn += f"Size {int(train_size * 100)}:{int(100 - (train_size*100))} = {prediction}\n"

        result_label_knn.config(text=result_text_knn)

        # Predict using model SVM
        predictions = {}
        for train_size, model in model_svm.items():
            predictions[train_size] = predict_image(file_path, model)

        # Display predictions SVM
        result_text_svm = "Predicted SVM:\n"
        for train_size, prediction in predictions.items():
            result_text_svm += f"Size {int(train_size * 100)}:{int(100 - (train_size*100))} = {prediction}\n"

        result_label_svm.config(text=result_text_svm)

        # Predict using model Decission Tree
        predictions = {}
        for train_size, model in model_dc.items():
            predictions[train_size] = predict_image(file_path, model)

        # Display predictions Decission Tree
        result_text_dc = "Predicted Decission Tree:\n"
        for train_size, prediction in predictions.items():
            result_text_dc += f"Size {int(train_size * 100)}:{int(100 - (train_size*100))} = {prediction}\n"

        result_label_dcTree.config(text=result_text_dc)


# Model KNN
model_paths_knn = {
    0.6: 'KNN_60.pkl',
    0.7: 'KNN_70.pkl',
    0.8: 'KNN_80.pkl'
}
model_knn = {}
for train_size, path in model_paths_knn.items():
    model_knn[train_size] = load_model(path)
print(model_knn)

# Model SVM
model_paths_svm = {
    0.6: 'SVM_60.pkl',
    0.7: 'SVM_70.pkl',
    0.8: 'SVM_80.pkl'
}
model_svm = {}
for train_size, path in model_paths_svm.items():
    model_svm[train_size] = load_model(path)
print(model_svm)

# Model Decission Tree
model_paths_dc = {
    0.6: 'DC_60.pkl',
    0.7: 'DC_70.pkl',
    0.8: 'DC_80.pkl'
}
model_dc = {}
for train_size, path in model_paths_dc.items():
    model_dc[train_size] = load_model(path)
print(model_dc)

# GUI Aplication
window = tk.Tk()
window.title("Sistem Pengenalan Digit")
window.geometry("650x800")
window.iconbitmap("bulb.ico")
window.configure(bg="#ffffff")

subtitle = tk.Label(window, text="Sistem Pengenalan Digit", font=(
    "Times New Roman", 25, 'bold'), fg="#000000", bg="#ffffff")
subtitle.pack(pady=5)

panel = tk.Label(window, bg="#ffffff")
panel.pack(padx=10, pady=10)

btn = tk.Button(window, text="Select Image", command=open_image, font=(
    "Times New Roman", 15), bg="#0000ff", fg="white", bd=0, padx=20, pady=10, borderwidth=0)
btn.pack(pady=10)

best_model = tk.Label(window, text="Best Model : \nSize 60:40 -- KNN: 89.17% SVM: 96.98% DC: 71.60%\nSize 70:30 --  KNN: 89.26% SVM: 97.17% DC: 73.18%\nSize 80:20 -- KNN: 89.60% SVM: 97.42% DC: 75.94%",
                      font=("Times New Roman", 16), bg="#ffffff", fg="#000000", justify=tk.CENTER)
best_model.pack(padx=10, pady=10)

result_label_knn = tk.Label(window, text="Predicted KNN: ", font=(
    "Times New Roman", 16), bg="#ffffff", fg="#000000", justify=tk.LEFT)
result_label_knn.pack(side='left', expand=True)

result_label_svm = tk.Label(window, text="Predicted SVM: ", font=(
    "Times New Roman", 16), bg="#ffffff", fg="#000000", justify=tk.LEFT)
result_label_svm.pack(side='left', expand=True)

result_label_dcTree = tk.Label(window, text="Predicted Decission Tree: ", font=(
    "Times New Roman", 16), bg="#ffffff", fg="#000000", justify=tk.LEFT)
result_label_dcTree.pack(side='left', expand=True)

window.mainloop()
