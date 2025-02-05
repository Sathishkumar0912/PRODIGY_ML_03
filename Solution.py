import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set your folder paths (UPDATE THESE PATHS)
train_dir = r"C:\Users\sathi\OneDrive\Desktop\WORKSPACE\Prodigy\Task-3\dogs-vs-cats\train\train"  
test_dir = r"C:\Users\sathi\OneDrive\Desktop\WORKSPACE\Prodigy\Task-3\dogs-vs-cats\test1\test1"  
output_csv = r"C:\Users\sathi\OneDrive\Desktop\WORKSPACE\Prodigy\Task-3\dogs-vs-cats\sampleSubmission.csv"  

# Debug: Check if folders exist
print(f"Checking paths...\nTrain folder: {train_dir}\nTest folder: {test_dir}\nCSV output path: {output_csv}")
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    print("Error: One or more directories do not exist. Check paths!")
    exit()

# Image Preprocessing
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    img = cv2.resize(img, (64, 64))  # Resize for consistency
    features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)  # HOG feature extraction
    return features

# Load training data (use a smaller sample for testing)
print("Loading training data... (this may take a while)")
X, y = [], []
for category in ["cat", "dog"]:
    for i in range(100):  # Change this number to load only 100 images for testing
        image_path = os.path.join(train_dir, f"{category}.{i}.jpg")
        if os.path.exists(image_path):
            features = extract_features(image_path)
            if features is not None:
                X.append(features)
                y.append(category)
        else:
            print(f"Skipping missing file: {image_path}")

print(f"Total training images loaded: {len(X)}")
X = np.array(X)
y = np.array(y)

# Encode labels
print("Encoding labels...")
le = LabelEncoder()
y = le.fit_transform(y)  # Cat -> 0, Dog -> 1

# Train-test split
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM Model
print("Training SVM model...")
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)
print("Training complete!")

# Evaluate Model
print("Evaluating model...")
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.4f}")

# Load test images
print("Loading test images...")
test_data = []
test_ids = []
for i in range(1, 101):  # Change this number for testing (use 100 images)
    image_path = os.path.join(test_dir, f"{i}.jpg")
    if os.path.exists(image_path):
        features = extract_features(image_path)
        if features is not None:
            test_data.append(features)
            test_ids.append(i)

print(f"Total test images loaded: {len(test_data)}")
test_data = np.array(test_data)

# Predict on test data
print("Making predictions on test images...")
predictions = svm.predict(test_data)
print(f"First 5 predictions: {predictions[:5]}")

# Save results to CSV
print(f"Saving predictions to {output_csv}...")
submission = pd.DataFrame({"id": test_ids, "label": predictions})
submission.to_csv(output_csv, index=False)
print("CSV saved successfully!")
