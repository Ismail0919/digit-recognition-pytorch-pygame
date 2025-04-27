import torch
import pygame
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

pygame.init()

# creating model class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
		# type of neural network
        self.flatten = nn.Flatten()
		# amount of neurons on each layer 784 -> 512 -> 512 -> 10
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# function that draws the paintings
def draw_paintings(paints):
	for i in range(len(paints)):
		pygame.draw.rect(screen, paints[i][0], (paints[i][1][0] - 10, paints[i][1][1] - 10, 20, 20))

device = "cpu"

# initialising the model and loading weight from created model
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

# list for the results
classes = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

# making the model work
model.eval()

# pygame work
screen = pygame.display.set_mode((280, 280))
painting = []
timer = pygame.time.Clock()
run = True
while run:
	screen.fill('black')
	timer.tick(60)
	mouse = pygame.mouse.get_pos()
	left_click = pygame.mouse.get_pressed()[0]
	# adding information if user press the LC on the mouse
	if left_click:
		painting.append(['white', mouse, 5])
	draw_paintings(painting)
	pygame.display.update()
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_SPACE or event.key == pygame.K_KP_ENTER:
				# taking screenshot of the pygame screen and saving it
				screenshot = pygame.Surface((280, 280))
				screenshot.blit(screen, (0, 0))
				screenshot = pygame.transform.scale(screenshot, (28, 28))
				pygame.image.save(screenshot, "test_image.png")
				# converting the saved screenshot to the Tensor object so we can feed it to the model
				test_image = cv2.imread("test_image.png", cv2.IMREAD_GRAYSCALE)
				transform = transforms.Compose([ToTensor()])
				x = transform(test_image)
				# getting the prediction from the model and outputing the result
				with torch.no_grad():
					x = x.to(device)
					pred = model(x)
					predicted = classes[pred[0].argmax(0)]
					print(f'Predicted: "{predicted}"')
				painting = []
				plt.imshow(test_image)
				plt.show()