import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier  # <-- Using the FAST classifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- 1. Function to Load a LIMITED sample of Images for SPEED ---
def load_images_from_folders(main_folder_path, max_images_per_class=1000):
    """
    Loads a limited number of images from subfolders for FASTER training.
    """
    images = []
    labels = []
    print(f"Starting to load a limited sample of {max_images_per_class} images per class...")
    
    for label_folder in ["Cat", "Dog"]:
        folder_path = os.path.join(main_folder_path, label_folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder '{folder_path}' not found. Skipping.")
            continue
        
        image_count = 0
        for image_file in os.listdir(folder_path):
            if image_count >= max_images_per_class:
                break # Stop once we have enough images for this class
            
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                labels.append(label_folder)
                image_count += 1

    print(f"Finished loading sample. Found {len(images)} images in total.")
    return images, labels

# --- 2. Function for Preprocessing and Feature Extraction ---
def preprocess_and_extract_features(images):
    """
    Resizes images, converts to grayscale, and extracts HOG features.
    """
    features = []
    total_images = len(images)
    print(f"Starting feature extraction from {total_images} images...")
    for i, image in enumerate(images):
        if (i + 1) % 100 == 0 or (i + 1) == total_images:
            print(f"  Processing image {i + 1}/{total_images}")
        resized_image = cv2.resize(image, (64, 128))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        features.append(hog_features)
    print("Feature extraction finished.")
    return np.array(features)

# --- 3. Main Script Execution ---
if __name__ == "__main__":
    dataset_path = 'PetImages'
    
    # --- Step 1: Load Data ---
    images, labels = load_images_from_folders(dataset_path)
    
    if not images:
        print("\nError: No images were loaded. Please check your folder structure.")
    else:
        # --- Step 2: Extract Features ---
        X = preprocess_and_extract_features(images)
        y = np.array(labels)
        
        # --- Step 3: Split Data and Train Model ---
        print("\nSplitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("\nTraining the model... (This will be very fast)")
        
        # Use the lightning-fast SGDClassifier to learn a linear SVM
        model = SGDClassifier(loss='hinge', random_state=42)
        
        model.fit(X_train, y_train)
        print("Model training complete.")
        
        # --- Step 4: Evaluate the Model ---
        print("\nEvaluating the model on the test set...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # --- Step 5: Save the Trained Model ---
        model_filename = 'svm_model.joblib'
        print(f"\nSaving the trained model to '{model_filename}'...")
        joblib.dump(model, model_filename)
        print("Model saved successfully!")
        print(f"\nYou can now use the '{model_filename}' file in your application.")