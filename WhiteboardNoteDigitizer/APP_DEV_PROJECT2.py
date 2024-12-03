# whiteboard_digitizer_v2.py
import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import logging
import os
from pathlib import Path

# Configure Tesseract path for Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class WhiteboardDigitizerPro:
    def __init__(self, root):
        self.root = root
        self.root.title("Whiteboard Digitizer Pro")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.cap = None
        self.camera_active = False
        self.captured_image = None
        self.processed_image = None
        self.detected_text = ""
        self.detected_shapes = []
        
        # Add window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('Custom.TFrame', background='#f0f0f0')
        self.style.configure('Custom.TButton', padding=5, font=('Helvetica', 10))
        self.style.configure('Title.TLabel', font=('Helvetica', 12, 'bold'))
        
        # Main container
        self.main_container = ttk.Frame(root, style='Custom.TFrame')
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create two main sections
        self.setup_input_section()
        self.setup_output_section()

    def setup_input_section(self):
        # Input section (left side)
        self.input_frame = ttk.Frame(self.main_container, style='Custom.TFrame')
        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        ttk.Label(self.input_frame, text="Input Section", style='Title.TLabel').pack(pady=10)
        
        self.preview_frame = ttk.Frame(self.input_frame, style='Custom.TFrame')
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack()
        
        controls_frame = ttk.Frame(self.input_frame, style='Custom.TFrame')
        controls_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(controls_frame, 
                  text="Upload Image",
                  command=self.upload_image,
                  style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        self.camera_btn = ttk.Button(controls_frame,
                                   text="Start Camera",
                                   command=self.toggle_camera,
                                   style='Custom.TButton')
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_frame,
                  text="Capture",
                  command=self.capture_image,
                  style='Custom.TButton').pack(side=tk.LEFT, padx=5)

    def setup_output_section(self):
        # Output section (right side)
        self.output_frame = ttk.Frame(self.main_container, style='Custom.TFrame')
        self.output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        ttk.Label(self.output_frame, text="Results Section", style='Title.TLabel').pack(pady=10)
        
        self.processed_preview = ttk.Label(self.output_frame)
        self.processed_preview.pack(pady=10)
        
        self.result_text = tk.Text(self.output_frame, 
                                 height=15, 
                                 width=50, 
                                 font=('Helvetica', 10))
        self.result_text.pack(pady=10)
        
        button_frame = ttk.Frame(self.output_frame, style='Custom.TFrame')
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame,
                  text="Process",
                  command=self.process_image,
                  style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame,
                  text="Export",
                  command=self.export_notes,
                  style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame,
                  text="Reset All",
                  command=self.reset_all,
                  style='Custom.TButton').pack(side=tk.LEFT, padx=5)

    def reset_all(self):
        self.captured_image = None
        self.processed_image = None
        self.detected_text = ""
        self.detected_shapes = []
        
        self.preview_label.configure(image='')
        self.processed_preview.configure(image='')
        self.result_text.delete(1.0, tk.END)
        
        if self.camera_active:
            self.toggle_camera()
        
        messagebox.showinfo("Reset", "All results have been cleared")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")])
        if file_path:
            self.captured_image = cv2.imread(file_path)
            self.update_preview(self.captured_image)

    def toggle_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.camera_active = True
                self.camera_btn.configure(text="Stop Camera")
                self.update_camera()
        else:
            self.camera_active = False
            self.camera_btn.configure(text="Start Camera")
            if self.cap:
                self.cap.release()
                self.cap = None

    def update_camera(self):
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.update_preview(frame)
                self.root.after(10, self.update_camera)

    def update_preview(self, frame):
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img.thumbnail((600, 400))
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)

    def capture_image(self):
        if self.cap and self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                self.captured_image = frame
                self.update_preview(frame)
                messagebox.showinfo("Success", "Image captured!")

    def process_image(self):
        if self.captured_image is None:
            messagebox.showerror("Error", "No image to process!")
            return
        
        try:
            # Enhanced image preprocessing for whiteboard
            image = self.captured_image.copy()
            
            # Convert to grayscale and apply bilateral filter
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Apply Otsu's thresholding with Gaussian blur
            blur = cv2.GaussianBlur(enhanced, (5,5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Deskew image
            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            (h, w) = thresh.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(thresh, M, (w, h), 
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            
            # Scale image for better OCR
            scaled = cv2.resize(rotated, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Apply text-specific preprocessing
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            dilated = cv2.dilate(scaled, kernel, iterations=1)
            
            # Enhanced OCR configuration
            custom_config = r'--oem 3 --psm 6 ' \
                        r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ' \
                        r'--dpi 300'
            
            try:
                self.detected_text = pytesseract.image_to_string(
                    dilated,
                    config=custom_config,
                    lang='eng'
                )
                # Post-process text
                self.detected_text = ' '.join(self.detected_text.split())
                
            except pytesseract.TesseractNotFoundError:
                messagebox.showerror("Error", "Tesseract is not installed or not found in PATH")
                return
            
            # Improved shape detection with the enhanced image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.detected_shapes = []
            min_area = 100
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    perimeter = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
                    
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w)/h
                        if 0.95 <= aspect_ratio <= 1.05:
                            shape = "square"
                        else:
                            shape = "rectangle"
                        self.detected_shapes.append((shape, approx, area))
                    elif len(approx) == 3:
                        self.detected_shapes.append(("triangle", approx, area))
                    elif len(approx) >= 8:
                        self.detected_shapes.append(("circle", approx, area))
            
            # Create annotated image with both shapes and text regions
            annotated_image = self.captured_image.copy()
            
            # Draw shape contours
            for shape, approx, area in self.detected_shapes:
                if shape == "square":
                    color = (0, 255, 0)
                elif shape == "rectangle":
                    color = (255, 0, 0)
                elif shape == "triangle":
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 0)
                cv2.drawContours(annotated_image, [approx], 0, color, 2)
            
            self.processed_image = annotated_image
            self.update_processed_preview(annotated_image)
            
            # Format results with improved layout
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=== Detected Text ===\n", "heading")
            self.result_text.insert(tk.END, f"{self.detected_text.strip()}\n\n")
            
            self.result_text.insert(tk.END, "=== Detected Shapes ===\n", "heading")
            sorted_shapes = sorted(self.detected_shapes, key=lambda x: x[2], reverse=True)
            for shape, _, area in sorted_shapes:
                self.result_text.insert(tk.END, f"• {shape.title()} (Area: {area:.0f} pixels)\n")
            
            self.result_text.tag_configure("heading", font=("Helvetica", 11, "bold"))
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error processing image: {str(e)}")

    def update_processed_preview(self, image):
        if image is not None:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            img = Image.fromarray(image)
            img.thumbnail((600, 400))
            imgtk = ImageTk.PhotoImage(image=img)
            self.processed_preview.imgtk = imgtk
            self.processed_preview.configure(image=imgtk)

    def export_notes(self):
        if self.detected_text == "":
            messagebox.showerror("Error", "No processed content to export!")
            return
        
        export_dir = filedialog.askdirectory()
        if export_dir:
            text_path = Path(export_dir) / "whiteboard_notes.txt"
            with open(text_path, "w") as f:
                f.write("=== Whiteboard Notes ===\n\n")
                f.write("Text Content:\n")
                f.write("-" * 40 + "\n")
                f.write(self.detected_text.strip() + "\n\n")
                f.write("Detected Shapes:\n")
                f.write("-" * 40 + "\n")
                sorted_shapes = sorted(self.detected_shapes, key=lambda x: x[2], reverse=True)
                for shape, _, area in sorted_shapes:
                    f.write(f"• {shape.title()} (Area: {area:.0f} pixels)\n")
            
            if self.processed_image is not None:
                img_path = Path(export_dir) / "whiteboard_image_annotated.png"
                cv2.imwrite(str(img_path), self.processed_image)
            
            messagebox.showinfo("Success", 
                              f"Notes exported successfully to {export_dir}")

    def cleanup(self):
        try:
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def on_closing(self):
        self.cleanup()
        self.root.destroy()

    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
    root = tk.Tk()
    app = WhiteboardDigitizerPro(root)
    root.mainloop()