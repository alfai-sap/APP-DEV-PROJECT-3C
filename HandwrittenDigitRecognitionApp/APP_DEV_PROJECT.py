import tkinter as tk
from tkinter import ttk, filedialog, Canvas
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition App")
        self.root.geometry("800x600")
        
        # Load the pre-trained model
        try:
            self.model = tf.keras.models.load_model('mnist_model.h5')
        except:
            print("Model file not found. Please ensure mnist_model.h5 exists.")
            self.model = None
            
        # Initialize screens and result label
        self.current_screen = None
        self.screens = {}
        self.result_label = None
        
        self.create_main_menu()
        self.create_draw_screen()
        self.create_upload_screen()
        self.show_screen("main_menu")
    
    def process_image(self, image):
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28
        image = cv2.resize(image, (28, 28))
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Reshape for model input
        image = image.reshape(1, 28, 28, 1)
        
        return image

    def predict_drawing(self):
        if self.model is None:
            self.show_result("Error: Model not loaded")
            return
            
        # Convert canvas to image
        x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
        y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        # Capture canvas content
        image = ImageTk.PhotoImage(self.canvas.postscript(file="canvas.eps"))
        image = cv2.imread("canvas.eps")
        
        if image is None:
            self.show_result("Error capturing drawing")
            return
            
        # Process image
        processed_image = self.process_image(image)
        
        # Make prediction
        prediction = self.model.predict(processed_image)
        digit = np.argmax(prediction[0])
        confidence = float(prediction[0][digit])
        
        self.show_result(f"Predicted: {digit} (Confidence: {confidence:.2%})")

    def upload_image(self):
        if self.model is None:
            self.show_result("Error: Model not loaded")
            return
            
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.eps")])
            
        if filename:
            try:
                image = cv2.imread(filename)
                processed_image = self.process_image(image)
                
                prediction = self.model.predict(processed_image)
                digit = np.argmax(prediction[0])
                confidence = float(prediction[0][digit])
                
                self.show_result(f"Predicted: {digit} (Confidence: {confidence:.2%})")
            except Exception as e:
                self.show_result(f"Error processing image: {str(e)}")

    def show_result(self, text):
        # Remove previous result if exists
        if self.result_label:
            self.result_label.destroy()
            
        # Create new result label
        self.result_label = tk.Label(self.current_screen, 
                                   text=text,
                                   font=('Arial', 16))
        self.result_label.pack(pady=10)

    def clear_canvas(self):
        self.canvas.delete("all")
        if self.result_label:
            self.result_label.destroy()
            self.result_label = None