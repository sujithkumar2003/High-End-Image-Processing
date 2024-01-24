import os
import cv2
import shutil
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Step 1: Image Preprocessing
def preprocess_image(image):
    # Preprocessing code
    new_width = image.shape[1]
    new_height = image.shape[0]
    resized_image = cv2.resize(image, (new_width, new_height))
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    normalized_image = cv2.normalize(blurred_image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image


# Step 2: Feature Extraction
def extract_features(image, max_feature_length):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a feature extraction algorithm
    feature_extractor = cv2.SIFT_create()
    keypoints, descriptors = feature_extractor.detectAndCompute(gray, None)

    # Check if descriptors exist
    if descriptors is None or len(descriptors) == 0:
        return None

    # Reshape descriptors to be a 1-dimensional array
    feature_vector = descriptors.flatten()

    # Check if feature vector length exceeds the maximum feature length
    if len(feature_vector) > max_feature_length:
        feature_vector = feature_vector[:max_feature_length]
    else:
        # Pad the feature vector if it is shorter than the maximum feature length
        feature_vector = np.pad(feature_vector, (0, max_feature_length - len(feature_vector)), mode='constant')

    return feature_vector


# Step 3: Image Registration with Robust Estimation
def image_registration(image1, image2):
    # Detect and match features
    src_points, dst_points = detect_and_match_features(image1, image2)

    # Perform robust estimation using RANSAC
    transformation_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    return transformation_matrix


# Step 4: Perspective Transformation
def perspective_transformation(image, transformation_matrix):
    # Perspective transformation code
    transformed_image = cv2.warpPerspective(image, transformation_matrix, (image.shape[1], image.shape[0]))
    return transformed_image


# Function for extracting features from an image
def extract_features(image, max_feature_length):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a feature extraction algorithm (e.g., SIFT, SURF, etc.)
    feature_extractor = cv2.SIFT_create()
    keypoints, descriptors = feature_extractor.detectAndCompute(gray, None)

    # Check if descriptors exist
    if descriptors is None or len(descriptors) == 0:
        return None

    # Reshape descriptors to be a 1-dimensional array
    feature_vector = descriptors.flatten()

    # Check if feature vector length exceeds the maximum feature length
    if len(feature_vector) > max_feature_length:
        feature_vector = feature_vector[:max_feature_length]
    else:
        # Pad the feature vector if it is shorter than the maximum feature length
        feature_vector = np.pad(feature_vector, (0, max_feature_length - len(feature_vector)), mode='constant')

    return feature_vector


# Function for detecting and matching features
def determine_max_feature_length(train_folder):
    max_feature_length = 0

    for filename in os.listdir(train_folder):
        if filename.endswith(".jpg"):
            train_image_path = os.path.join(train_folder, filename)
            train_image = cv2.imread(train_image_path)
            processed_train_image = preprocess_image(train_image, new_width, new_height)

            feature_vector = extract_features(processed_train_image, 128)
            if feature_vector is not None and len(feature_vector) > max_feature_length:
                max_feature_length = len(feature_vector)

    return max_feature_length


# Function for finding the ROI in the images
def find_roi(image):
    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold to obtain a binary image.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the binary image.
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour that is similar to the brand name.
    largest_cnt = None
    max_area = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > max_area and area > 1000:
            largest_cnt = cnt
            max_area = area

    # Get the bounding box of the largest contour.
    if largest_cnt is not None:
        x, y, w, h = cv2.boundingRect(largest_cnt)

        # Return the bounding box.
        return x, y, w, h
    else:
        return None

# Step 5: Complete Image Registration Pipeline
def register_images(train_folder, validation_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_images = []
    train_features = []
    train_labels = []

    for folder_name in ['authentic', 'counterfeit']:
        folder_path = os.path.join(train_folder, folder_name)

        if not os.path.exists(folder_path):
            continue

        label = 1 if folder_name == 'authentic' else 0  # Assign label 1 for 'authentic' folder and 0 for 'counterfeit' folder

        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(folder_path, filename)
                train_image = cv2.imread(image_path)

                # Find the ROI for the current train image
                x, y, w, h = find_roi(train_image)

                if x is not None:
                    # Crop the ROI from the image
                    roi = train_image[y:y + h, x:x + w]

                    # Preprocess the ROI
                    processed_roi = preprocess_image(roi)

                    train_images.append(processed_roi)

                    feature_vector = extract_features(processed_roi, 10)  # Replace max_feature_length with your desired value
                    if feature_vector is not None:
                        train_features.append(feature_vector)
                        train_labels.append(label)
                    else:
                        print("No features found for image:", filename)
                else:
                    print("ROI not found for image:", filename)

    # Check if any train features were extracted
    if len(train_features) == 0:
        print("No train features found.")
        return 0, 0, 0

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    # Apply clustering to train features and obtain cluster centers
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(train_features)
    cluster_centers = kmeans.cluster_centers_

    # Train the classifier with the extracted features
    base_classifier = RandomForestClassifier()
    bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10)  # Change the number of estimators as needed
    bagging_classifier.fit(train_features, train_labels)

    # Move processed validation images to final validation folder
    final_validation_folder = os.path.join(output_folder, "validation")
    if not os.path.exists(final_validation_folder):
        os.makedirs(final_validation_folder)

    validation_images = []
    validation_features = []

    for filename in os.listdir(validation_folder):
        if filename.endswith(".jpg"):
            validation_image_path = os.path.join(validation_folder, filename)
            validation_image = cv2.imread(validation_image_path)

            # Find the ROI for the current validation image
            x, y, w, h = find_roi(validation_image)

            if x is not None:
                # Crop the ROI from the image
                roi = validation_image[y:y + h, x:x + w]

                # Preprocess the ROI
                processed_roi = preprocess_image(roi)

                validation_images.append(processed_roi)

                feature_vector = extract_features(processed_roi, 10)  # Replace max_feature_length with your desired value
                if feature_vector is not None:
                    validation_features.append(feature_vector)
                else:
                    print("No features found for image:", filename)
            else:
                print("ROI not found for image:", filename)

    # Check if any validation features were extracted
    if len(validation_features) == 0:
        print("No validation features found.")
        return 0, 0, 0

    validation_features = np.array(validation_features)

    # Transform validation features to match the cluster centers
    transformed_validation_features = kmeans.transform(validation_features)

    # Make predictions on the transformed validation features
    validation_predictions = bagging_classifier.predict(transformed_validation_features)

    # Move the validation images to the corresponding output folders based on the predictions
    for i, prediction in enumerate(validation_predictions):
        filename = os.listdir(validation_folder)[i]
        source_path = os.path.join(validation_folder, filename)
        destination_folder = os.path.join(final_validation_folder, "authentic" if prediction == 1 else "counterfeit")
        destination_path = os.path.join(destination_folder, filename)

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        shutil.move(source_path, destination_path)

    return len(train_images), len(validation_images), len(validation_predictions)




def main():
  
    train_folder = r"C:\Users\mikip\OneDrive\Desktop\Ennovate\DataSet\Train_Images"  # Path to your train folder 
    validation_folder = r"C:\Users\mikip\OneDrive\Desktop\Ennovate\DataSet\Valid_Images"   # Path to your valid folder
    output_folder = r"C:\Users\mikip\OneDrive\Desktop\Ennovate\output"    # Path to your output folder (where you want to save the final_validation)

    accuracy, precision, recall = register_images(train_folder, validation_folder, output_folder)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)


if __name__ == '__main__':
    main()
# Open a new terminal and type : python .\Final_Code.py    
# The scores after training the model were: 
# Accuracy: 52
# Precision: 3
# Recall: 3

