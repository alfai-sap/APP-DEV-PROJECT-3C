import os
import cv2
import numpy as np  # Fix the import statement
import pytesseract
from googletrans import Translator
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageGrab, ImageOps  # Add ImageOps for inverting colors
import logging
from pathlib import Path
import time
from gtts import gTTS
import pygame
import requests
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import wikitextparser as wtp
import html
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def check_tesseract_installation():
    """Check if Tesseract is properly installed and configured"""
    if not os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        raise RuntimeError("Tesseract is not installed in the expected location")
    
    tessdata_path = r'C:\Program Files\Tesseract-OCR\tessdata'
    if not os.path.exists(tessdata_path):
        raise RuntimeError("Tessdata directory not found")
    
    if not os.path.exists(os.path.join(tessdata_path, 'eng.traineddata')):
        raise RuntimeError("English language data not found")

# Configure Tesseract path
if os.name == 'nt':  # Windows
    try:
        check_tesseract_installation()
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
        logging.info("Tesseract configured successfully")
    except Exception as e:
        logging.error(f"Tesseract configuration error: {e}")
        messagebox.showerror("Error", 
            "Tesseract is not properly installed.\n"
            "Please install Tesseract-OCR and ensure the language data files are present.")
        sys.exit(1)

def enhance_image(image):
    """Enhanced image processing pipeline specifically for handwriting recognition"""
    try:
        # Add padding to the image
        padding = 20
        height, width = image.shape[:2]
        padded_image = cv2.copyMakeBorder(
            image,
            padding, padding, padding, padding,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        
        # Convert to grayscale
        gray = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10
        )
        
        # Noise removal and text enhancement
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Dilate to make text more prominent
        binary = cv2.dilate(binary, kernel, iterations=1)
        
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
        self.pen_width = 3  # Restore original pen width
        self.eraser_width = 50
        
        # Add real-time processing variables
        self.real_time_active = False
        self.last_process_time = time.time()
        self.process_delay = 1.0  # 1 second delay
        
        self.stroke_completed = False
        self.last_stroke_time = 0
        self.stroke_delay = 0.5  # Delay after stroke completion before detection

        # Update the languages dictionary with correct codes
        # Use 'eng' for Tesseract but 'en' for Google Translate
        self.languages = {
            'English': {'ocr': 'eng', 'translate': 'en'},
            'Filipino': {'ocr': 'tgl', 'translate': 'tl'},  # Add Filipino/Tagalog
            'Cebuano': {'ocr': 'ceb', 'translate': 'ceb'},  # Add Cebuano/Bisaya
            'Spanish': {'ocr': 'spa', 'translate': 'es'},
            'French': {'ocr': 'fra', 'translate': 'fr'},
            'German': {'ocr': 'deu', 'translate': 'de'},
            'Chinese': {'ocr': 'chi_sim', 'translate': 'zh-cn'},
            'Japanese': {'ocr': 'jpn', 'translate': 'ja'},
            'Italian': {'ocr': 'ita', 'translate': 'it'},
            'Portuguese': {'ocr': 'por', 'translate': 'pt'},
            'Russian': {'ocr': 'rus', 'translate': 'ru'},
            'Korean': {'ocr': 'kor', 'translate': 'ko'},
            'Arabic': {'ocr': 'ara', 'translate': 'ar'},
            'Dutch': {'ocr': 'nld', 'translate': 'nl'},
            'Greek': {'ocr': 'ell', 'translate': 'el'},
            'Hindi': {'ocr': 'hin', 'translate': 'hi'},
            'Turkish': {'ocr': 'tur', 'translate': 'tr'},
            'Vietnamese': {'ocr': 'vie', 'translate': 'vi'},
            'Thai': {'ocr': 'tha', 'translate': 'th'},
            'Polish': {'ocr': 'pol', 'translate': 'pl'},
            'Indonesian': {'ocr': 'ind', 'translate': 'id'},
            'Swedish': {'ocr': 'swe', 'translate': 'sv'}
        }
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Add LibreTranslate URL
        self.libretranslate_url = "https://libretranslate.com/translate"
        
        # Initialize thread pool for concurrent API calls
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Add audio state tracking
        self.current_audio = None
        self.audio_playing = False
        
        # Create temp directory for audio files
        self.temp_audio_dir = os.path.join(tempfile.gettempdir(), 'handwriting_app_audio')
        os.makedirs(self.temp_audio_dir, exist_ok=True)
        
        # Initialize pygame mixer with better settings
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        
        self.setup_gui()

    def setup_gui(self):
        # Main container
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel (Input)
        left_panel = ttk.Frame(container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(
            left_panel,
            width=800,
            height=400,
            bg='white',
            highlightthickness=1,
            highlightbackground="gray"
        )
        self.canvas.pack(pady=10, padx=10)
        
        # Update canvas bindings
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stroke_completed)
        self.canvas.bind("<Button-1>", self.start_stroke)
        
        # Controls
        controls = ttk.Frame(left_panel)
        controls.pack(fill=tk.X, pady=10)
        
            # Language selection with swap button
        lang_frame = ttk.LabelFrame(left_panel, text="Languages")
        lang_frame.pack(fill=tk.X, pady=10)
        
        lang_controls = ttk.Frame(lang_frame)
        lang_controls.pack(fill=tk.X, padx=5, pady=5)
        
        # Source language
        ttk.Label(lang_controls, text="From:").pack(side=tk.LEFT, padx=5)
        self.source_lang = ttk.Combobox(lang_controls, 
                                    values=sorted(list(self.languages.keys())), 
                                    state='readonly',
                                    width=15)
        self.source_lang.pack(side=tk.LEFT, padx=5)
        self.source_lang.set("English")
        
        # Swap languages button
        ttk.Button(lang_controls, 
                text="⇄",
                command=self.swap_languages).pack(side=tk.LEFT, padx=5)
        
        # Target language
        ttk.Label(lang_controls, text="To:").pack(side=tk.LEFT, padx=5)
        self.target_lang = ttk.Combobox(lang_controls,
                                    values=sorted(list(self.languages.keys())),
                                    state='readonly',
                                    width=15)
        self.target_lang.pack(side=tk.LEFT, padx=5)
        self.target_lang.set("Spanish")
        
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
        
        # Remove duplicate language selection frame
        # Keep only the one inside the controls section
        
        # Right panel (Output)
        right_panel = ttk.Frame(container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Recognized Text Section
        recognized_frame = ttk.LabelFrame(right_panel, text="Recognized Text", padding="10")
        recognized_frame.pack(fill=tk.X, pady=(5, 10), padx=5)
        
        self.recognized_text_label = ttk.Label(
            recognized_frame,
            text="No text recognized yet",
            wraplength=400,
            justify=tk.LEFT,
            style="Result.TLabel"
        )
        self.recognized_text_label.pack(fill=tk.X, pady=5)
        
        # Translation Section with Pronunciation and Description
        translation_frame = ttk.LabelFrame(right_panel, text="Translation", padding="10")
        translation_frame.pack(fill=tk.X, pady=(5, 10), padx=5)
        
        self.translated_text_label = ttk.Label(
            translation_frame,
            text="No translation available",
            wraplength=400,
            justify=tk.LEFT,
            style="Result.TLabel"
        )
        self.translated_text_label.pack(fill=tk.X, pady=5)
        
        # Add pronunciation button
        self.pronunciation_button = ttk.Button(
            translation_frame,
            text="🔊 Listen",
            command=self.play_pronunciation
        )
        self.pronunciation_button.pack(pady=5)
        
        # Description Section
        description_frame = ttk.LabelFrame(right_panel, text="Description & Pronunciation Guide", padding="10")
        description_frame.pack(fill=tk.X, pady=(5, 10), padx=5)
        
        # Replace Text widget with Label
        self.description_label = ttk.Label(
            description_frame,
            text="No description available",
            wraplength=400,
            justify=tk.LEFT,
            style="Result.TLabel"
        )
        self.description_label.pack(fill=tk.X, pady=5)

        # Create custom styles for the labels
        style = ttk.Style()
        style.configure(
            "Result.TLabel",
            font=('Arial', 12),
            background='#f0f0f0',
            padding=10
        )

    def swap_languages(self):
        """Swap source and target languages"""
        source = self.source_lang.get()
        target = self.target_lang.get()
        self.source_lang.set(target)
        self.target_lang.set(source)

    def update_tool(self):
        self.current_tool = self.tool_var.get()
        self.status_label.config(text=f"Tool: {self.current_tool.title()}")

    def toggle_realtime(self):
        self.real_time_active = self.realtime_var.get()
        if self.real_time_active:
            self.process_real_time()

    def process_real_time(self):
        if self.real_time_active and self.stroke_completed:
            current_time = time.time()
            if current_time - self.last_stroke_time >= self.stroke_delay:
                self.recognize_text(real_time=True)
                self.stroke_completed = False  # Reset for next stroke
            if self.real_time_active:
                self.root.after(100, self.process_real_time)
    
    def start_stroke(self, event):
        self.stroke_completed = False
        self.last_x = event.x
        self.last_y = event.y

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
        self.recognized_text_label.config(text="No text recognized yet")
        self.translated_text_label.config(text="No translation available")
        self.description_label.config(text="No description available")

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
            # Get the exact canvas coordinates
            canvas_x = self.canvas.winfo_rootx()
            canvas_y = self.canvas.winfo_rooty()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Add padding to capture area
            padding = 10
            image = ImageGrab.grab(bbox=(
                canvas_x - padding,
                canvas_y - padding,
                canvas_x + canvas_width + padding,
                canvas_y + canvas_height + padding
            ))
            
            # Convert to numpy array and process
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process image
            processed = enhance_image(cv_image)
            if processed is None:
                raise Exception("Image processing failed")
            
            # Update OCR configuration for better edge detection
            custom_config = (
                '--oem 1 '  # LSTM OCR Engine
                '--psm 6 '  # Assume uniform block of text
                '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
                '-c tessedit_write_images=1 '
                '-c preserve_interword_spaces=1 '
                '--dpi 300'  # Increase DPI for better recognition
            )
            
            # Perform OCR with increased page segmentation
            text = pytesseract.image_to_string(
                processed,
                config=custom_config,
                timeout=5
            )
            text = ' '.join(text.strip().split())  # Clean up whitespace
            
            if not text.strip():
                if not real_time:
                    messagebox.showinfo("Info", "No text detected")
                return
            
            # Update UI
            self.recognized_text_label.config(text=text)
            
            # Handle translation
            if text.strip():
                try:
                    source_lang = self.source_lang.get()
                    target_lang = self.target_lang.get()
                    
                    # Perform translation
                    translation = self.translator.translate(
                        text,
                        src=self.languages[source_lang]['translate'],
                        dest=self.languages[target_lang]['translate']
                    )
                    
                    # Update translation display
                    if translation and translation.text:
                        self.translated_text_label.config(text=translation.text)
                        
                        # Get pronunciation guide
                        target_lang_code = self.languages[target_lang]['translate']
                        pronunciation = self.get_pronunciation_guide(translation.text, target_lang_code)
                        
                        # Get word descriptions
                        words = translation.text.split()
                        descriptions = []
                        
                        for word in words[:3]:
                            desc = self.get_word_description(word, target_lang_code)
                            if desc:
                                descriptions.append(f"\n{word}:\n{desc}")
                        
                        # Update description label
                        full_text = f"Language: {target_lang}\n"
                        if pronunciation:
                            full_text += f"{pronunciation}\n"
                        if descriptions:
                            full_text += "\nDefinitions:" + "".join(descriptions)
                        else:
                            full_text += "\nNo detailed definitions available."
                        
                        self.description_label.config(text=full_text)
                        
                    else:
                        logging.warning("Translation returned empty result")
                        if not real_time:
                            messagebox.showwarning("Warning", "Translation failed")
                            
                except Exception as e:
                    logging.error(f"Translation error: {str(e)}")
                    if not real_time:
                        messagebox.showerror("Translation Error", 
                                           f"Failed to translate text: {str(e)}")
                    self.translated_text_label.config(text="Translation error occurred")
                    
        except Exception as e:
            logging.error(f"Recognition error: {e}")
            if not real_time:
                messagebox.showerror("Error", str(e))

    def stroke_completed(self, event):
        self.stroke_completed = True
        self.last_stroke_time = time.time()
        self.reset_coordinates(event)
        
        if self.real_time_active:
            current_time = time.time()
            if current_time - self.last_process_time >= self.process_delay:
                self.recognize_text(real_time=True)
                self.last_process_time = current_time

    def play_pronunciation(self):
        """Play the pronunciation of the translated text with improved error handling"""
        try:
            text = self.translated_text_label.cget("text")
            if text and text != "No translation available":
                # Stop any currently playing audio
                if self.audio_playing:
                    pygame.mixer.music.stop()
                    self.audio_playing = False
                
                # Generate unique filename
                temp_file = os.path.join(
                    self.temp_audio_dir,
                    f"pronunciation_{uuid.uuid4().hex[:8]}.mp3"
                )
                
                # Clean up old files
                self.cleanup_old_audio_files()
                
                try:
                    # Generate speech
                    tts = gTTS(
                        text=text,
                        lang=self.languages[self.target_lang.get()]['translate'],
                        slow=False
                    )
                    
                    # Save with error handling
                    for _ in range(3):  # Retry up to 3 times
                        try:
                            tts.save(temp_file)
                            break
                        except PermissionError:
                            time.sleep(0.1)  # Wait briefly before retry
                    
                    # Load and play the audio
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        pygame.mixer.music.load(temp_file)
                        pygame.mixer.music.play()
                        self.audio_playing = True
                        self.current_audio = temp_file
                        
                        # Schedule cleanup
                        self.root.after(100, self.check_audio_finished)
                    else:
                        raise Exception("Failed to generate audio file")
                        
                except Exception as e:
                    logging.error(f"Audio generation error: {e}")
                    self.cleanup_audio_file(temp_file)
                    raise
                    
        except Exception as e:
            logging.error(f"Pronunciation error: {e}")
            messagebox.showerror("Error", "Failed to play pronunciation. Please try again.")
    
    def check_audio_finished(self):
        """Check if audio has finished playing and clean up"""
        if self.audio_playing and not pygame.mixer.music.get_busy():
            self.audio_playing = False
            if self.current_audio:
                self.cleanup_audio_file(self.current_audio)
                self.current_audio = None
        elif self.audio_playing:
            # Check again in 100ms if still playing
            self.root.after(100, self.check_audio_finished)
    
    def cleanup_audio_file(self, filepath):
        """Safely clean up an audio file"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logging.warning(f"Failed to cleanup audio file {filepath}: {e}")
    
    def cleanup_old_audio_files(self):
        """Clean up old audio files from temp directory"""
        try:
            current_time = time.time()
            for filename in os.listdir(self.temp_audio_dir):
                filepath = os.path.join(self.temp_audio_dir, filename)
                # Remove files older than 1 hour
                if os.path.getctime(filepath) < current_time - 3600:
                    self.cleanup_audio_file(filepath)
        except Exception as e:
            logging.warning(f"Failed to cleanup old audio files: {e}")
    
    def __del__(self):
        """Cleanup on application exit"""
        try:
            # Stop any playing audio
            if hasattr(self, 'audio_playing') and self.audio_playing:
                pygame.mixer.music.stop()
            
            # Clean up temp directory
            if hasattr(self, 'temp_audio_dir') and os.path.exists(self.temp_audio_dir):
                for filename in os.listdir(self.temp_audio_dir):
                    self.cleanup_audio_file(os.path.join(self.temp_audio_dir, filename))
                try:
                    os.rmdir(self.temp_audio_dir)
                except:
                    pass
        except:
            pass

    def get_word_description(self, word, lang_code):
        """Enhanced word description using free APIs"""
        descriptions = []
        
        # Try Free Dictionary API first
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/{lang_code}/{quote(word.lower())}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    meanings = data[0].get('meanings', [])
                    if meanings:
                        for meaning in meanings[:2]:
                            pos = meaning.get('partOfSpeech', '')
                            definition = meaning.get('definitions', [{}])[0].get('definition', '')
                            if definition:
                                descriptions.append(f"({pos}) {definition}")
                                # Add example if available
                                example = meaning.get('definitions', [{}])[0].get('example', '')
                                if example:
                                    descriptions.append(f"   Example: {example}")
        except Exception as e:
            logging.error(f"Free Dictionary API error: {e}")

        # Try Wiktionary API as backup
        if not descriptions:
            try:
                url = f"https://en.wiktionary.org/w/api.php"
                params = {
                    'action': 'parse',
                    'format': 'json',
                    'page': word.lower(),
                    'prop': 'wikitext',
                    'section': 0
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if 'parse' in data and 'wikitext' in data['parse']:
                        wikitext = data['parse']['wikitext']['*']
                        parsed = wtp.parse(wikitext)
                        # Extract definitions from wikitext
                        for section in parsed.sections:
                            if 'Etymology' in section.title or 'Definitions' in section.title:
                                clean_text = html.unescape(section.plain_text())
                                if clean_text:
                                    descriptions.append(clean_text[:200] + "...")  # Limit length
                                break
            except Exception as e:
                logging.error(f"Wiktionary API error: {e}")

        # Try MyMemory API as final fallback
        if not descriptions:
            try:
                url = f"https://api.mymemory.translated.net/get?q={quote(word)}&langpair={lang_code}|en"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('matches'):
                        for match in data['matches'][:2]:
                            if match.get('translation') and match.get('quality', '0') != '0':
                                descriptions.append(f"≈ {match['translation']}")
            except Exception as e:
                logging.error(f"MyMemory API error: {e}")

        return '\n'.join(descriptions) if descriptions else None

    def get_pronunciation_guide(self, text, lang_code):
        """Get pronunciation guide using LibreTranslate"""
        try:
            # Try to get IPA pronunciation from Wiktionary
            url = f"https://en.wiktionary.org/w/api.php"
            params = {
                'action': 'parse',
                'format': 'json',
                'page': text.lower(),
                'prop': 'wikitext',
                'section': 0
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'parse' in data and 'wikitext' in data['parse']:
                    wikitext = data['parse']['wikitext']['*']
                    # Look for IPA pronunciation
                    if '{{IPA|' in wikitext:
                        ipa_start = wikitext.find('{{IPA|') + 6
                        ipa_end = wikitext.find('}}', ipa_start)
                        if ipa_end > ipa_start:
                            return f"IPA: {wikitext[ipa_start:ipa_end]}"
            
            # Fallback to simplified pronunciation guide
            return f"Pronunciation available through 'Listen' button"
            
        except Exception as e:
            logging.error(f"Pronunciation guide error: {e}")
            return None

def main():
    root = tk.Tk()
    app = MultilingualRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()