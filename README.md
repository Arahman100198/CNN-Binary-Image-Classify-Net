# Image Classification with CNN (Keras + TensorFlow)
# CNN-Binary-Image-Classify-Net
This project demonstrates an image classification model built using a Convolutional Neural Network (CNN) with Keras and TensorFlow. The goal of the project is to classify images using a CNN architecture with regularization techniques like L2 Regularization and Dropout to enhance the model's performance and reduce overfitting.

## Description
The model architecture consists of several layers:

1- Convolutional layers to extract features from the images.
2- MaxPooling layers to reduce the dimensionality of feature maps.
3- Batch Normalization to normalize outputs and accelerate training.
4- Dropout layers to randomly drop connections between layers and prevent overfitting.
5- Fully Connected layers for final image classification.
6- An Exponential Learning Rate Decay and EarlyStopping are applied to prevent overfitting and improve model convergence.

## Key Features:

1- L2 Regularization: Applied to prevent large weights.
2- Dropout Layers: Used to reduce overfitting by randomly setting some weights to zero.
3- Early Stopping: Stops training if validation loss does not improve for a specified number of epochs.
4- Learning Rate Decay: Reduces the learning rate progressively for better model convergence.

## Findings and Results

1- The model shows a significant reduction in training loss and an increase in accuracy over multiple epochs.
2- Validation accuracy and loss fluctuated slightly, but overall performance improved with the use of regularization techniques.
3- Dropout and L2 Regularization helped reduce overfitting, allowing the model to generalize better on unseen data.
4- The EarlyStopping callback was effective in preventing overtraining and halting at the optimal point, ensuring the best model performance.

## Conclusion

This project highlights the effectiveness of CNNs, L2 Regularization, and Dropout layers for image classification tasks. The use of these techniques helps improve generalization, reduces overfitting, and results in a robust model for classification tasks. Future improvements could include experimenting with different architectures and tuning hyperparameters for even better performance.
