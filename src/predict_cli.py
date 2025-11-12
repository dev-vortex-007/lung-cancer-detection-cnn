import os
import sys
import argparse
import numpy as np
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt

class SimpleLungCancerPredictor:
    def __init__(self, model_path):
        # Class labels
        self.class_names = ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']
        self.img_size = (400, 400)
        
        # Load model
        print("Loading model...")
        try:
            self.model = load_model(model_path)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
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
    
    def predict(self, img_path, show_image=False):
        """Make prediction on a single image"""
        print(f"\nAnalyzing image: {img_path}")
        
        # Preprocess image
        processed_img = self.preprocess_image(img_path)
        if processed_img is None:
            return None
        
        # Make prediction
        print("Making prediction...")
        predictions = self.model.predict(processed_img, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        
        # Display results
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Predicted Class: {predicted_class.replace('_', ' ').title()}")
        print(f"Confidence: {confidence:.2%}")
        print("\nAll Class Probabilities:")
        print("-" * 30)
        
        for i, class_name in enumerate(self.class_names):
            prob = predictions[0][i]
            formatted_name = class_name.replace('_', ' ').title()
            print(f"{formatted_name:<25}: {prob:.2%}")
        
        print("="*50)
        
        # Show image if requested
        if show_image:
            self.display_image(img_path)
        
        return predicted_class, confidence, predictions[0]
    
    def display_image(self, img_path):
        """Display the image using matplotlib"""
        try:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(img_rgb)
            plt.title(f"CT Scan Image: {os.path.basename(img_path)}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")
    
    def predict_batch(self, image_directory):
        """Predict on all images in a directory"""
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(image_directory) 
                      if f.lower().endswith(supported_formats)]
        
        if not image_files:
            print(f"No supported image files found in {image_directory}")
            return
        
        print(f"\nFound {len(image_files)} images to analyze...")
        results = []
        
        for img_file in image_files:
            img_path = os.path.join(image_directory, img_file)
            result = self.predict(img_path)
            if result:
                results.append((img_file, result[0], result[1]))
        
        # Summary
        print("\n" + "="*70)
        print("BATCH PREDICTION SUMMARY")
        print("="*70)
        for img_file, pred_class, confidence in results:
            formatted_class = pred_class.replace('_', ' ').title()
            print(f"{img_file:<30} | {formatted_class:<25} | {confidence:.2%}")
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Lung Cancer Detection Prediction Tool')
    parser.add_argument('--image', '-i', type=str, help='Path to single image for prediction')
    parser.add_argument('--directory', '-d', type=str, help='Path to directory containing images')
    parser.add_argument('--model', '-m', type=str, help='Path to model file')
    parser.add_argument('--show-image', '-s', action='store_true', help='Display the image')
    
    args = parser.parse_args()
    
    # Default model path
    if not args.model:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, '../models/cnn_model.keras')
        if not os.path.exists(model_path):
            model_path = os.path.join(script_dir, '../models/cnn_model.h5')
    else:
        model_path = args.model
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first or specify the correct model path with --model")
        sys.exit(1)
    
    # Initialize predictor
    predictor = SimpleLungCancerPredictor(model_path)
    
    if args.image:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            sys.exit(1)
        predictor.predict(args.image, args.show_image)
    
    elif args.directory:
        # Batch prediction
        if not os.path.exists(args.directory):
            print(f"Directory not found: {args.directory}")
            sys.exit(1)
        predictor.predict_batch(args.directory)
    
    else:
        # Interactive mode
        print("Lung Cancer Detection - Interactive Mode")
        print("="*40)
        
        while True:
            print("\nOptions:")
            print("1. Analyze single image")
            print("2. Analyze directory of images")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                img_path = input("Enter path to image: ").strip()
                if os.path.exists(img_path):
                    show_img = input("Show image? (y/n): ").strip().lower() == 'y'
                    predictor.predict(img_path, show_img)
                else:
                    print("Image file not found!")
            
            elif choice == '2':
                dir_path = input("Enter path to directory: ").strip()
                if os.path.exists(dir_path):
                    predictor.predict_batch(dir_path)
                else:
                    print("Directory not found!")
            
            elif choice == '3':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()