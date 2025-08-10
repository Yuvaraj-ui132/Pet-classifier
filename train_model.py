import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # A simple, fast classifier is enough now
from sklearn.metrics import accuracy_score
import joblib

# --- 1. Load the Pre-trained Deep Learning Model (MobileNetV2) ---
# We use the 'include_top=False' to remove the final classification layer,
# so we can use the model for feature extraction.
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False # We freeze the model's weights

# --- 2. Function to Load and Preprocess Images ---
def load_and_preprocess_images(folder_path, max_images_per_class=1000):
    images = []
    labels = []
    print(f"Loading a sample of {max_images_per_class} images per class...")
    for label in ["Cat", "Dog"]:
        class_path = os.path.join(folder_path, label)
        if not os.path.isdir(class_path): continue
        
        count = 0
        for filename in os.listdir(class_path):
            if count >= max_images_per_class: break
            
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Preprocess for MobileNetV2: resize to 224x224 and use its specific function
                img = cv2.resize(img, (224, 224))
                img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img)
                images.append(img_preprocessed)
                labels.append(label)
                count += 1
    
    print(f"Loaded and preprocessed {len(images)} images.")
    return np.array(images), np.array(labels)

# --- 3. Main Script ---
if __name__ == "__main__":
    dataset_path = 'PetImages'
    
    # Load data
    images, labels = load_and_preprocess_images(dataset_path)
    
    if len(images) > 0:
        # Extract features using the deep learning model
        print("Extracting features with MobileNetV2...")
        features = base_model.predict(images)
        # Flatten the features from 4D to 2D
        num_samples = features.shape[0]
        flattened_features = features.reshape(num_samples, -1)
        print("Feature extraction complete.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(flattened_features, labels, test_size=0.2, random_state=42, stratify=labels)
        
        # Train a simple classifier on the powerful features
        print("Training the final classifier...")
        # Logistic Regression is fast and works great with strong features
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X_train, y_train)
        print("Training complete.")
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
        
        # Save the new model
        model_filename = 'deep_learning_model.joblib'
        print(f"Saving new model to '{model_filename}'...")
        joblib.dump(classifier, model_filename)
        # We also need to save the class names
        class_names = list(classifier.classes_)
        joblib.dump(class_names, 'class_names.joblib')
        print("Model saved successfully!")