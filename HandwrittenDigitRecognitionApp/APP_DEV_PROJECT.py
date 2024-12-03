# APP_DEV_PROJECT.py
import os
import cv2
import numpy as np
import pytesseract
from googletrans import Translator
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageGrab
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure Tesseract path
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Update the enhance_image function
def enhance_image(image):
    """Enhanced image processing pipeline for handwriting recognition"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logging.debug("Converted to grayscale")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        logging.debug("Applied Gaussian blur")
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        logging.debug("Enhanced contrast")
        
        # Thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, 
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        logging.debug("Applied thresholding")
        
        # Noise removal
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        logging.debug("Removed noise")
        
        # Dilation to connect components
        binary = cv2.dilate(binary, kernel, iterations=1)
        logging.debug("Applied dilation")
        
        return binary
    except Exception as e:
        logging.error(f"Image enhancement failed: {e}")
        return None

class MultilingualRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multilingual Handwriting Recognition")
        self.root.geometry("1400x800")
        
        # Initialize variables
        self.translator = Translator()
        self.last_x = None
        self.last_y = None
        self.captured_image = None
        self.real_time_active = False
        self.last_process_time = 0
        self.process_delay = 1.0
        
        # Add new variables for tools
        self.current_tool = "pen"
        self.pen_color = "black"
        self.eraser_color = "white"
        self.pen_width = 3
        self.eraser_width = 50
        
        # Add real-time processing variables
        self.real_time_active = False
        self.last_process_time = time.time()
        self.process_delay = 1.0  # 1 second delay
        
        # Languages
        self.languages = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Chinese': 'zh-cn',
            'Japanese': 'ja',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Russian': 'ru',
            'Korean': 'ko',
            'Arabic': 'ar',
            'Dutch': 'nl',
            'Greek': 'el',
            'Hindi': 'hi',
            'Turkish': 'tr',
            'Vietnamese': 'vi',
            'Thai': 'th',
            'Polish': 'pl',
            'Indonesian': 'id',
            'Swedish': 'sv'
        }
        
        self.setup_gui()

    def setup_gui(self):
        # Main container
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel (Input)
        left_panel = ttk.Frame(container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(left_panel, width=800, height=400, bg='white')
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coordinates)
        
        # Controls
        controls = ttk.Frame(left_panel)
        controls.pack(fill=tk.X, pady=10)
        
        # Tool controls
        self.tool_var = tk.StringVar(value="pen")
        ttk.Radiobutton(controls, text="Pen", variable=self.tool_var, 
                       value="pen", command=self.update_tool).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(controls, text="Eraser", variable=self.tool_var,
                       value="eraser", command=self.update_tool).pack(side=tk.LEFT, padx=5)
        
        # Real-time toggle
        self.realtime_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Real-time Recognition",
                       variable=self.realtime_var,
                       command=self.toggle_realtime).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Upload", command=self.upload_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Recognize", command=self.recognize_text).pack(side=tk.LEFT, padx=5)
        
        # Status indicator
        self.status_label = ttk.Label(controls, text="Tool: Pen")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Language selection
        lang_frame = ttk.LabelFrame(left_panel, text="Languages")
        lang_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(lang_frame, text="From:").pack(side=tk.LEFT, padx=5)
        self.source_lang = ttk.Combobox(lang_frame, values=sorted(list(self.languages.keys())), state='readonly')
        self.source_lang.pack(side=tk.LEFT, padx=5)
        self.source_lang.set("English")
        
        ttk.Label(lang_frame, text="To:").pack(side=tk.LEFT, padx=5)
        self.target_lang = ttk.Combobox(lang_frame, values=sorted(list(self.languages.keys())), state='readonly')
        self.target_lang.pack(side=tk.LEFT, padx=5)
        self.target_lang.set("Spanish")
        
        # Right panel (Output)
        right_panel = ttk.Frame(container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_panel, text="Recognized Text:").pack(pady=5)
        self.recognized_text = tk.Text(right_panel, height=10, width=50)
        self.recognized_text.pack(pady=5)
        
        ttk.Label(right_panel, text="Translation:").pack(pady=5)
        self.translated_text = tk.Text(right_panel, height=10, width=50)
        self.translated_text.pack(pady=5)

    def update_tool(self):
        self.current_tool = self.tool_var.get()
        self.status_label.config(text=f"Tool: {self.current_tool.title()}")

    def toggle_realtime(self):
        self.real_time_active = self.realtime_var.get()
        if self.real_time_active:
            self.process_real_time()

    def process_real_time(self):
        if self.real_time_active:
            current_time = time.time()
            if current_time - self.last_process_time >= self.process_delay:
                self.recognize_text(real_time=True)
                self.last_process_time = current_time
            self.root.after(100, self.process_real_time)

    def paint(self, event):
        if self.last_x and self.last_y:
            color = self.eraser_color if self.current_tool == "eraser" else self.pen_color
            width = self.eraser_width if self.current_tool == "eraser" else self.pen_width
            
            self.canvas.create_line(
                self.last_x, self.last_y,
                event.x, event.y,
                width=width,
                fill=color,
                capstyle=tk.ROUND,
                smooth=tk.TRUE
            )
        self.last_x = event.x
        self.last_y = event.y
        
        if self.real_time_active:
            current_time = time.time()
            if current_time - self.last_process_time >= self.process_delay:
                self.recognize_text(real_time=True)
                self.last_process_time = current_time

    def reset_coordinates(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.recognized_text.delete(1.0, tk.END)
        self.translated_text.delete(1.0, tk.END)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")])
        if file_path:
            self.captured_image = cv2.imread(file_path)
            self.display_image()

    def display_image(self):
        if self.captured_image is not None:
            image = cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image.thumbnail((800, 400))
            photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo

    def recognize_text(self, real_time=False):
        try:
            # Capture canvas
            x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
            y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
            x1 = x + self.canvas.winfo_width()
            y1 = y + self.canvas.winfo_height()
            
            image = ImageGrab.grab(bbox=(x, y, x1, y1))
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process image
            processed = enhance_image(cv_image)
            if processed is None:
                raise Exception("Image processing failed")
            
            # Save processed image for debugging
            cv2.imwrite('processed.png', processed)
            logging.debug("Saved processed image")
            
            # Configure Tesseract for handwriting
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
            
            # OCR
            text = pytesseract.image_to_string(
                processed,
                config=custom_config,
                lang='eng'
            )
            logging.debug(f"OCR Result: {text}")
            
            if not text.strip():
                messagebox.showinfo("Info", "No text detected")
                return
            
            if not real_time:
                self.recognized_text.delete(1.0, tk.END)
            else:
                self.recognized_text.delete(1.0, tk.END)  # Clear previous real-time results
            
            self.recognized_text.insert(tk.END, text + "\n")
            
            # Translate
            if text.strip():
                source_lang = self.source_lang.get()
                target_lang = self.target_lang.get()
                
                try:
                    translation = self.translator.translate(
                        text,
                        src=self.languages[source_lang],
                        dest=self.languages[target_lang]
                    )
                    
                    if not real_time:
                        self.translated_text.delete(1.0, tk.END)
                    else:
                        self.translated_text.delete(1.0, tk.END)  # Clear previous real-time results
                    
                    self.translated_text.insert(tk.END, translation.text + "\n")
                    
                except Exception as e:
                    logging.error(f"Translation error: {e}")
                    if not real_time:
                        messagebox.showerror("Translation Error", 
                                           "Failed to translate text")
        
        except Exception as e:
            logging.error(f"Recognition error: {e}")
            if not real_time:
                messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = MultilingualRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()