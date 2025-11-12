import os
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

class LungCancerPredictor:
    def __init__(self):
        # Class labels
        self.class_names = ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']
        self.model = None
        self.img_size = (400, 400)
        
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = load_model(model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, img_path):
        """Preprocess image for prediction"""
        try:
            # Load and resize image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Could not load image")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values to [0, 1]
            img = img.astype('float32') / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, img_path):
        """Make prediction on a single image"""
        if self.model is None:
            return None, None
        
        # Preprocess image
        processed_img = self.preprocess_image(img_path)
        if processed_img is None:
            return None, None
        
        # Make prediction
        predictions = self.model.predict(processed_img, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {
            self.class_names[i]: float(predictions[0][i]) 
            for i in range(len(self.class_names))
        }
        
        return predicted_class, confidence, class_probabilities

class LungCancerDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Cancer Detection System")
        self.root.geometry("1100x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize predictor
        self.predictor = LungCancerPredictor()
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Variables
        self.selected_image_path = tk.StringVar()
        self.prediction_result = tk.StringVar()
        self.confidence_score = tk.StringVar()
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Lung Cancer Detection System", 
            font=('Arial', 20, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Subtitle
        subtitle_label = ttk.Label(
            main_frame, 
            text="Upload a chest CT scan image for analysis", 
            font=('Arial', 12)
        )
        subtitle_label.pack(pady=(0, 20))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        # File path display
        path_frame = ttk.Frame(file_frame)
        path_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(path_frame, text="Selected Image:").pack(anchor=tk.W)
        path_entry = ttk.Entry(path_frame, textvariable=self.selected_image_path, state='readonly')
        path_entry.pack(fill=tk.X, pady=(5, 0))
        
        # Buttons frame
        button_frame = ttk.Frame(file_frame)
        button_frame.pack(fill=tk.X)
        
        select_btn = ttk.Button(
            button_frame, 
            text="Select Image", 
            command=self.select_image,
            style='Accent.TButton'
        )
        select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        predict_btn = ttk.Button(
            button_frame, 
            text="Analyze Image", 
            command=self.predict_image,
            style='Accent.TButton'
        )
        predict_btn.pack(side=tk.LEFT)
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.image_label = ttk.Label(image_frame, text="No image selected", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas and scrollbar for scrollable results
        canvas = tk.Canvas(results_frame, bg='white')
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Result display
        result_display_frame = ttk.Frame(self.scrollable_frame)
        result_display_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(result_display_frame, text="Primary Diagnosis:", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        prediction_label = ttk.Label(
            result_display_frame, 
            textvariable=self.prediction_result, 
            font=('Arial', 16, 'bold'),
            foreground='blue'
        )
        prediction_label.pack(anchor=tk.W)
        
        ttk.Label(result_display_frame, text="Confidence Score:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(15, 0))
        confidence_label = ttk.Label(
            result_display_frame, 
            textvariable=self.confidence_score, 
            font=('Arial', 14, 'bold'),
            foreground='green'
        )
        confidence_label.pack(anchor=tk.W)
        
        # Detailed probabilities frame
        self.probabilities_frame = ttk.Frame(self.scrollable_frame)
        self.probabilities_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Progress bar
        self.progress = ttk.Progressbar(results_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def load_model(self):
        """Load the trained model"""
        model_path = os.path.join(os.path.dirname(__file__), '../models/cnn_model.keras')
        if not os.path.exists(model_path):
            # Try alternative path
            model_path = os.path.join(os.path.dirname(__file__), '../models/cnn_model.h5')
        
        if os.path.exists(model_path):
            if self.predictor.load_model(model_path):
                self.status_var.set("Model loaded successfully")
            else:
                self.status_var.set("Error loading model")
                messagebox.showerror("Error", "Failed to load the trained model")
        else:
            self.status_var.set("Model not found - Please train the model first")
            messagebox.showwarning("Warning", "Model file not found. Please train the model first.")
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select CT Scan Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_image_path.set(file_path)
            self.display_image(file_path)
            self.prediction_result.set("")
            self.confidence_score.set("")
            # Clear previous detailed results
            for widget in self.probabilities_frame.winfo_children():
                widget.destroy()
    
    def display_image(self, image_path):
        """Display the selected image in the GUI"""
        try:
            # Open and resize image for display
            pil_image = Image.open(image_path)
            
            # Calculate size to fit in the display area while maintaining aspect ratio
            display_size = (300, 300)
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update the label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {e}")
    
    def predict_image(self):
        """Make prediction on the selected image"""
        if not self.selected_image_path.get():
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        if self.predictor.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        # Start prediction in a separate thread to avoid freezing the GUI
        threading.Thread(target=self._predict_worker, daemon=True).start()
    
    def _predict_worker(self):
        """Worker function for prediction (runs in separate thread)"""
        try:
            # Update UI to show processing
            self.root.after(0, lambda: self.progress.start())
            self.root.after(0, lambda: self.status_var.set("Analyzing image..."))
            
            # Make prediction
            result = self.predictor.predict(self.selected_image_path.get())
            
            if result[0] is not None:
                predicted_class, confidence, class_probabilities = result
                
                # Update UI with results
                self.root.after(0, lambda: self.update_results(predicted_class, confidence, class_probabilities))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to analyze image"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Prediction failed: {e}"))
        finally:
            # Stop progress bar
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.status_var.set("Ready"))
    
    def update_results(self, predicted_class, confidence, class_probabilities):
        """Update the GUI with prediction results"""
        # Format the prediction result
        formatted_class = predicted_class.replace('_', ' ').title()
        self.prediction_result.set(formatted_class)
        self.confidence_score.set(f"{confidence:.2%}")
        
        # Clear previous detailed results
        for widget in self.probabilities_frame.winfo_children():
            widget.destroy()
        
        # Show detailed results in the main window
        self.show_detailed_probabilities(class_probabilities)
    
    def show_detailed_probabilities(self, class_probabilities):
        """Show detailed prediction probabilities in the main window"""
        # Title for probabilities section
        title_label = ttk.Label(
            self.probabilities_frame, 
            text="All Class Probabilities:", 
            font=('Arial', 12, 'bold')
        )
        title_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Create a separator
        separator = ttk.Separator(self.probabilities_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=(0, 10))
        
        # Sort probabilities by confidence (highest first)
        sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Color mapping for different types
        color_map = {
            'normal': '#2ECC71',           # Green for normal
            'adenocarcinoma': '#E74C3C',   # Red for cancer types
            'large_cell_carcinoma': '#E67E22',  # Orange for cancer types
            'squamous_cell_carcinoma': '#9B59B6'  # Purple for cancer types
        }
        
        for i, (class_name, prob) in enumerate(sorted_probs):
            formatted_name = class_name.replace('_', ' ').title()
            
            # Create frame for each probability
            prob_frame = ttk.Frame(self.probabilities_frame)
            prob_frame.pack(fill=tk.X, pady=3)
            
            # Class name label
            name_label = ttk.Label(
                prob_frame, 
                text=f"{i+1}. {formatted_name}:", 
                font=('Arial', 11, 'bold'),
                width=25
            )
            name_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Probability percentage
            prob_label = ttk.Label(
                prob_frame, 
                text=f"{prob:.2%}", 
                font=('Arial', 11, 'bold'),
                foreground=color_map.get(class_name, '#34495E')
            )
            prob_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Progress bar to visualize probability
            progress_bar = ttk.Progressbar(
                prob_frame, 
                length=200, 
                mode='determinate', 
                value=prob*100
            )
            progress_bar.pack(side=tk.LEFT, padx=(0, 10))
            
            # Confidence level indicator
            if prob > 0.7:
                confidence_text = "High"
                confidence_color = "green"
            elif prob > 0.4:
                confidence_text = "Medium"
                confidence_color = "orange"
            else:
                confidence_text = "Low"
                confidence_color = "gray"
            
            confidence_label = ttk.Label(
                prob_frame, 
                text=confidence_text, 
                font=('Arial', 9),
                foreground=confidence_color
            )
            confidence_label.pack(side=tk.LEFT)
        
        # Add interpretation section
        self.add_interpretation_section(sorted_probs[0])
    
    def add_interpretation_section(self, top_prediction):
        """Add interpretation section to help users understand the results"""
        class_name, probability = top_prediction
        
        # Separator
        separator = ttk.Separator(self.probabilities_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=(20, 10))
        
        # Interpretation title
        interpretation_label = ttk.Label(
            self.probabilities_frame,
            text="Interpretation:",
            font=('Arial', 12, 'bold')
        )
        interpretation_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Generate interpretation text
        formatted_class = class_name.replace('_', ' ').title()
        
        if class_name == 'normal':
            if probability > 0.8:
                interpretation = f"The scan appears to be normal with high confidence ({probability:.1%}). No signs of lung cancer detected."
            elif probability > 0.6:
                interpretation = f"The scan likely appears normal ({probability:.1%}), but consider additional medical evaluation."
            else:
                interpretation = f"Results are uncertain. Professional medical evaluation is strongly recommended."
        else:
            if probability > 0.8:
                interpretation = f"Strong indication of {formatted_class} ({probability:.1%}). Immediate medical consultation recommended."
            elif probability > 0.6:
                interpretation = f"Possible indication of {formatted_class} ({probability:.1%}). Medical evaluation advised."
            else:
                interpretation = f"Uncertain results with slight indication of {formatted_class}. Further testing recommended."
        
        # Interpretation text
        interpretation_text = tk.Text(
            self.probabilities_frame,
            height=3,
            wrap=tk.WORD,
            font=('Arial', 10),
            bg='#F8F9FA',
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        interpretation_text.pack(fill=tk.X, pady=(0, 10))
        interpretation_text.insert('1.0', interpretation)
        interpretation_text.config(state=tk.DISABLED)
        
        # Disclaimer
        disclaimer_text = tk.Text(
            self.probabilities_frame,
            height=2,
            wrap=tk.WORD,
            font=('Arial', 9),
            bg='#FFF3CD',
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        disclaimer_text.pack(fill=tk.X, pady=(5, 0))
        disclaimer_text.insert('1.0', "⚠️ Disclaimer: This AI tool is for educational purposes only. Always consult qualified medical professionals for proper diagnosis and treatment.")
        disclaimer_text.config(state=tk.DISABLED)

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = LungCancerDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
