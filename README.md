# High End Image Registration and Authentication 📷✨

This project aims to enhance image registration accuracy by integrating feature extraction and machine learning techniques. The goal is to achieve precise alignment between images, even when significant features are missing. The project utilizes the 🌐 OpenCV library for image preprocessing, feature detection and matching, and perspective transformation. It also employs machine learning algorithms 🤖 for feature extraction and image classification.


## Features 🚀

- **Image Preprocessing:** The images undergo preprocessing techniques like resizing, blurring, and normalization to improve quality and consistency.

- **Feature Extraction:** Key feature points are detected and descriptors are extracted using algorithms like SIFT. These descriptors are then transformed into feature vectors.

- **Robust Estimation:** Robust estimation methods, such as RANSAC, are utilized to obtain accurate transformation matrices for aligning the images.

- **Perspective Transformation:** The images are transformed based on the estimated transformation matrices to achieve precise alignment.

- **Feature Clustering:** KMeans clustering is applied to the extracted features to group them into compact representations, improving efficiency and reducing dimensionality.

- **Machine Learning Classification:** The project employs a Random Forest classifier trained on the clustered features to classify images into authentic or counterfeit categories.


## Usage 🛠️

1. Install the required dependencies specified in the `requirements.txt` file.

2. Prepare your dataset by organizing authentic and counterfeit images in separate folders under the Train folder. 

3. Modify the file paths in the code to point to your dataset and output/final folders.

4. Run the `main()` function to execute the image registration pipeline and evaluate the performance. Run python .\<filename.py.


## Evaluation Metrics 📊

The project measures performance using the following metrics:

- **Accuracy:** The overall accuracy of the image classification model in correctly predicting the authenticity of images.

- **Precision:** The proportion of correctly classified authentic images to the total number of authentic predictions.

- **Recall:** The proportion of correctly classified authentic images to the total number of actual authentic images.


## Dataset 📁

The project utilizes a provided dataset containing authentic and counterfeit images for training and a combination of both in the valid folder for validation purposes.



## Contributions and feedback are welcome! 🙌

