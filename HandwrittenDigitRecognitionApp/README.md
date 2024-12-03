# APP-DEV-PROJECT-3C
HANDWRITTEN RECOGNITION APP

APP DEV PROJECT GUIDE

1. Setting Up the Environment and Libraries

Task: Set up the development environment, install libraries.

Steps:

Install Python (version 3.x) and an IDE like VS Code or PyCharm.

Install necessary libraries using pip:

pip install 
opencv-python-headless numpy
tensorflow pillow tk

Verify installations by testing imports in Python:

import cv2
import numpy as np
import tensorflow as tf
from tkinter import *


Assigned to: Both can set up their own environments initially. One person can focus on dependencies while the other starts with UI design.



2. Build the Tkinter GUI

Task: Create the GUI layout in Tkinter.

Steps:

Set up the main Tkinter window with the title, dimensions, and basic layout.

Create three main screens in the window:

1. Main Menu: Buttons for "Draw Digit" and "Upload Image".

2. Draw Canvas: A white canvas area where users can draw digits.

3. Upload Image Screen: A file selection option for image upload.



Add Buttons for switching between screens and a Label to display prediction results.


Assigned to: Teammate 1 works on the layout and canvas drawing functionality, while Teammate 2 sets up the file upload screen and button functionality.



3. Implement Digit Input Options

Task: Create options for drawing and uploading digits.

Steps:

Drawing:

Use Tkinter Canvas for drawing. Track mouse movement events to draw on the canvas.

Include Clear and Predict buttons.

Save the canvas drawing as an image (e.g., .png) in grayscale to process later.


Image Upload:

Use tkinter.filedialog to allow users to select an image file.

Convert the uploaded image to grayscale for processing.


Assigned to: Teammate 1 can handle canvas drawing and image saving, while Teammate 2 takes care of the image upload and conversion.


4. Pre-trained Model and AI Integration

Task: Load a pre-trained model and use it for digit prediction.

Steps:

Download a pre-trained MNIST model (trained on handwritten digits).

Load the model using TensorFlow or Keras:

from tensorflow.keras.models import load_model
model = load_model('mnist_model.h5')

Process the drawn or uploaded images:

Resize images to 28x28 pixels, convert to grayscale, and normalize pixel values.

Predict the digit using:

prediction = model.predict(image)
predicted_digit = np.argmax(prediction)


Assigned to: Teammate 1 works on loading the pre-trained model and image processing (resizing and normalization).



5. Connect AI Model with GUI for Predictions

Task: Link the AI model to the GUI for real-time predictions.

Steps:

Add prediction functions to the Predict button on both input screens (draw canvas and upload image).

Display the predicted digit result in a Tkinter Label.

If necessary, add loading or progress indicators for the prediction process.


Assigned to: Teammate 2 links the model prediction to GUI and formats the output display.



6. Real-Time Feedback and Testing

Task: Test and refine the real-time prediction.

Steps:

Test predictions for both input methods (drawing and image upload).

Adjust processing code for better performance if needed, ensuring results appear almost instantly.

Handle any potential errors in image processing or prediction and provide user-friendly messages.


Assigned to: Both teammates test on their setups and refine as needed.


7. Package the App as an Executable (.exe)

Task: Use Auto Py to Exe to create a standalone executable file.

Steps:

Install Auto Py to Exe:

pip install auto-py-to-exe

Launch Auto Py to Exe:

auto-py-to-exe

Configure settings:

Select the main .py file.

Set options to create a single .exe file.

Run to generate the .exe file in the specified directory.


Assigned to: Teammate 1 can handle this task.



8. Create an Installer with Inno Setup

Task: Use Inno Setup to create an installer.

Steps:

Download and install Inno Setup.

Open Inno Setup and create a new installer script.

Set up the script to include the .exe file generated and any additional dependencies.

Compile the script to create an installer .exe for the app.


Assigned to: Teammate 2 can handle this task.




9. Documentation and Final Testing

Task: Write brief documentation and perform final tests.

Steps:

Include a description of the app’s features, installation instructions, and usage guide.

Run the installer on a clean system to test the installation process and app functionality.

Assigned to: Both teammates can collaborate on documentation and final testing.


10. Submit the Project

Task: Gather the following items for submission:

Source Code: Clean and organize the Python files.

Executable File (.exe): The standalone app.

Installer: Created with Inno Setup.

Documentation: Include setup, usage, and troubleshooting information.

