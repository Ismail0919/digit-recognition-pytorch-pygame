# digit-recognition-pytorch-pygame
This project implements a digital number recognition system using neural networks. The system allows users to draw digits on a canvas and recognizes the drawn digit using a neural network model built with PyTorch.

Technologies Used:

PyGame: Used to create the drawing interface where users can draw digits.
PyTorch: Utilized to build and train the neural network model for digit recognition.
Conversion Tensor: The drawn images are converted to tensor objects to be processed by the neural network.

Requirements
Before running the application, ensure you have the following dependencies installed:
torch
pygame
numpy
cv2
matplotlib
torchvision

You can install them using pip:
pip install torch pygame numpy cv2 matplotlib torchvision

How to Use
Clone the Repository:
git clone https://github.com/Ismail0919/digit-recognition-pytorch-pygame.git

Create the Model:
First, you need to create the neural network model by running:
python model_creation.py

Run the Application: After creating the model, start the digit recognition application by running:
python main.py

How It Works
Drawing: Users can draw digits using the PyGame interface.
Recognition: The drawn digit is converted into a tensor and passed through the neural network model.
Output: The model predicts and displays the recognized digit.
