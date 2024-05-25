# ASL Recognition using Convolutional Neural Networks (CNN)

## Project Overview
This project focuses on the development of a machine learning model using Convolutional Neural Networks (CNN) to recognize American Sign Language (ASL) gestures. We collected and processed a large dataset of hand gestures to train and test our model.

## Dataset
The dataset includes hand gesture images representing five letters (A, B, C, D, E) in ASL:
- **Total Images:** 2,686 images
- **Average Images per Letter:** 470
- **Training Set:** 2,321 images
- **Testing Set:** 365 images

### Data Collection
Each image captures a specific hand gesture with annotated hand points that provide details about hand position and shape.

### Data Preprocessing
- **Resizing:** Images were resized to a standardized scale to ensure uniformity.
- **Scaling:** Pixel values were scaled between 0 and 1 by dividing each by 255.
- **Cropping:** Unwanted areas were cropped out to enhance the dataset quality.

## Model Architecture
We employed a CNN architecture designed specifically for ASL recognition, taking advantage of its ability to capture spatial hierarchies in images.

## Training
- **Model Compilation:** The model was compiled and trained on the processed dataset.
- **Training Accuracy:** 97%

## Evaluation
- **Test Accuracy:** 95%

## Performance
The accuracy and loss metrics were monitored over each epoch during training, showing significant convergence indicative of a successful learning process.
