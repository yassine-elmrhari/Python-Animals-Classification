# Python-Animals-Classification


## Description
This project focuses on image classification using the CIFAR-10 dataset. Two different approaches are explored:
1. **Machine Learning (ML) using feature extraction techniques and classification algorithms**
2. **Deep Learning (DL) using Convolutional Neural Networks (CNNs)**

The goal is to compare the performance of both approaches and determine their effectiveness in classifying images.


## Dataset
The CIFAR-10 dataset consists of 60,000 images (32x32 pixels in RGB) categorized into 10 classes with 6,000 images per class. There are 50,000 training images and 10,000 test images.


## Project phases
I. **Machine Learning Approach**  
  
  This phase involves feature extraction and classification using traditional ML algorithms.
  
  Steps:
  
  - Extract image features using:
  
    -- Hu Moments (Shape descriptor)
  
    -- Haralick Features (Texture descriptor)
  
    -- Color Histogram (Color descriptor)
  
  - Train ML classifiers including:
  
    -- K-Nearest Neighbors (KNN)
  
    -- Support Vector Machine (SVM)
  
    -- Random Forest (RF)
  
    -- Linear Discriminant Analysis (LDA)
  
    -- Stacking Classifier (Combining multiple classifiers)
  
  - Evaluate and compare classification accuracy.
<br></br>

II. **Deep Learning Approach**

  This phase applies Convolutional Neural Networks (CNNs) for classification.
  
  Model Architecture:
  
  - VGG-inspired CNN with:
  
    -- Multiple convolutional layers
  
    -- Batch normalization
  
    -- Dropout for regularization
  
    -- Fully connected layers for classification
  
  - Trained using Adam optimizer and categorical cross-entropy loss function.
  
  - Data augmentation techniques (flipping, shifting, zooming) are applied to improve performance.


## Implementation

- Preprocess CIFAR-10 images.

- Train and evaluate ML models.

- Train CNN with 300 epochs.

- Compare results between ML and DL approaches.


## Results

- Machine Learning Approach Accuracy: ~46.74% (best model: Stacking Classifier with Logistic Regression)

- Deep Learning Approach Accuracy: ~78% (after 300 epochs with augmentation and optimization)

- Observation: CNN significantly outperforms traditional ML algorithms.


## Conclusions

- Deep Learning provides superior accuracy compared to traditional Machine Learning for image classification.

- ML models can be useful for small datasets but struggle with complex image features.

- CNNs, especially with data augmentation and optimization techniques, significantly enhance performance.

  
## Requirements
- **Python 3.x**
- **NumPy 1.21.0 or higher**
- **OpenCV-Python 4.5.3 or higher**
- **Mahotas 1.4.11 or higher**
- **Scikit-Learn 1.0 or higher**
- **Matplotlib 3.4.3 or higher**
- **Pillow 8.4.0 or higher**
- **TensorFlow 2.6.0 or higher**
- **Keras 2.6.0 or higher**


## Installation and Execution
1. Clone this repository:
   ```sh
   git clone https://github.com/yassine-elmrhari/Python-Animals-Classification.git
   cd Python-Animals-Classification
   ```
2. Install necessary dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Train the Machine Learning Models:
   ```sh
   python TrainingAlgorithms.py
   ```
4. Train the Neural Network:
   ```sh
   python TrainingNeuralNetwork.py
   ```
5. Run the GUI Application:
   ```sh
   python ProjectGui.py
   ```


## References
- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Scikit-learn documentation: https://scikit-learn.org/
- TensorFlow documentation: https://www.tensorflow.org/


## Author
Yassine Elmrhari
