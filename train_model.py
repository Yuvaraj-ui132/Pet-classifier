import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import joblib

def load_full_dataset(main_folder_path):
    images, labels = [], []
    print(f"Starting to load FULL dataset from '{main_folder_path}'...")
    for label_folder in ["Cat", "Dog"]:
        folder_path = os.path.join(main_folder_path, label_folder)
        if not os.path.isdir(folder_path): continue
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                labels.append(label_folder)
    print(f"Finished loading. Found {len(images)} images.")
    return images, labels

def preprocess_and_extract_features(images):
    features = []
    total = len(images)
    print(f"Extracting HOG features from {total} images...")
    for i, image in enumerate(images):
        if (i + 1) % 500 == 0: print(f"  Processing image {i + 1}/{total}")
        resized_image = cv2.resize(image, (64, 128))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys')
        features.append(hog_features)
    print("Feature extraction finished.")
    return np.array(features)

if __name__ == "__main__":
    dataset_path = 'PetImages'
    images, labels = load_full_dataset(dataset_path)
    if images:
        X = preprocess_and_extract_features(images)
        y = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"\nTraining classifier on {len(X_train)} images... This will take a long time.")
        base_model = SGDClassifier(loss='hinge', random_state=42)
        model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        model.fit(X_train, y_train)
        print("Model training complete.")
        print("\nEvaluating model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
        model_filename = 'svm_model.joblib'
        print(f"\nSaving final model to '{model_filename}'...")
        joblib.dump(model, model_filename)
        print("Model saved successfully!")