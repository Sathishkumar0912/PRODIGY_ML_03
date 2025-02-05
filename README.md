# Cats vs Dogs Image Classifier

This repository contains a machine learning model that classifies images of cats and dogs using a Support Vector Machine (SVM) classifier with Histogram of Oriented Gradients (HOG) feature extraction. The model is trained on a dataset of images and can be used to predict whether an image contains a cat or a dog.

## Project Overview

- **Machine Learning Algorithm**: Support Vector Machine (SVM) with a linear kernel.
- **Feature Extraction Method**: Histogram of Oriented Gradients (HOG).
- **Dataset**: The dataset contains images of cats and dogs. The model is trained using the images in the `train` directory and evaluated on the images in the `test` directory.
- **Output**: The model outputs predictions in a CSV file, containing the image IDs and their corresponding labels (cat or dog).

## Directory Structure

```
.
├── train/                # Training dataset (cat and dog images)
│   ├── cat/              # Cat images
│   └── dog/              # Dog images
├── test1/                # Test dataset (images for prediction)
├── dogs-vs-cats.ipynb    # Jupyter Notebook for training and prediction (optional)
├── sampleSubmission.csv  # CSV file for output predictions
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```
## Usage

1. **Train the Model**:  
   The model can be trained by running the Python script. It will read images from the `train` directory, preprocess them, extract HOG features, and train an SVM model.

   ```
   python dogs-vs-cats.py
   ```

2. **Evaluate the Model**:  
   After training, the model will be evaluated on the test data and the accuracy will be displayed.

3. **Make Predictions**:  
   The trained model will predict labels for images in the `test1` directory. The predictions will be saved to a CSV file (`sampleSubmission.csv`), with the image ID and the predicted label (0 for cat, 1 for dog).

## Code Walkthrough

- **extract_features()**: This function extracts HOG features from an image and resizes it to a standard size.
- **Training**: The SVM classifier is trained using the HOG features extracted from the training images.
- **Prediction**: The trained model is used to predict labels for unseen test images.
- **Saving Output**: Predictions are saved in a CSV file containing the image IDs and the predicted labels.

## Acknowledgments

- Dataset: The [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle.
- Libraries: OpenCV, Scikit-learn, Scikit-image, Pandas.

### Notes:
1. **Installation**: The `requirements.txt` file is not provided in the code, so make sure to create it with the necessary dependencies.
2. **Paths**: Don’t forget to adjust paths in the code to match the correct folder structure on your machine or provide instructions to the users to update paths.
3. **Dataset**: The dataset used (e.g., Dogs vs. Cats from Kaggle) should be mentioned, and the user may need to download it separately. 

Let me know if you'd like to modify anything!
