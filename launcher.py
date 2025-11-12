#!/usr/bin/env python3
"""
Lung Cancer Detection System Launcher
=====================================

This script provides an easy way to launch either the GUI or CLI version
of the lung cancer detection system.

Usage:
    python launcher.py          # Interactive launcher
    python launcher.py --gui    # Launch GUI directly
    python launcher.py --cli    # Launch CLI directly
"""

import os
import sys
import argparse
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['tensorflow', 'opencv-python', 'numpy', 'matplotlib', 'pillow']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install them using:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_model():
    """Check if trained model exists"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths = [
        os.path.join(script_dir, 'models/cnn_model.keras'),
        os.path.join(script_dir, 'models/cnn_model.h5')
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            return True, model_path
    
    return False, None

def launch_gui():
    """Launch the GUI version"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gui_script = os.path.join(script_dir, 'src/predict.py')
        
        print("🚀 Launching GUI...")
        subprocess.run([sys.executable, gui_script])
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")

def launch_cli():
    """Launch the CLI version"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cli_script = os.path.join(script_dir, 'src/predict_cli.py')
        
        print("🚀 Launching CLI...")
        subprocess.run([sys.executable, cli_script])
    except Exception as e:
        print(f"❌ Failed to launch CLI: {e}")

def interactive_launcher():
    """Interactive launcher menu"""
    print("🔬 Lung Cancer Detection System")
    print("=" * 35)
    
    # Check requirements
    print("Checking system requirements...")
    if not check_requirements():
        return
    print("✅ All required packages are installed")
    
    # Check model
    model_exists, model_path = check_model()
    if not model_exists:
        print("⚠️  No trained model found!")
        print("Please train the model first by running:")
        print("   python src/train.py")
        
        train_now = input("\nWould you like to train the model now? (y/n): ").strip().lower()
        if train_now == 'y':
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                train_script = os.path.join(script_dir, 'train.py')
                print("🏋️  Starting training process...")
                subprocess.run([sys.executable, train_script])
                
                # Check again after training
                model_exists, model_path = check_model()
                if not model_exists:
                    print("❌ Training failed or model not saved properly")
                    return
            except Exception as e:
                print(f"❌ Training failed: {e}")
                return
        else:
            return
    
    print(f"✅ Model found: {os.path.basename(model_path)}")
    
    print("\nChoose your interface:")
    print("1. 🖥️  GUI (Graphical User Interface) - Recommended")
    print("2. 💻 CLI (Command Line Interface)")
    print("3. ❌ Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            launch_gui()
            break
        elif choice == '2':
            launch_cli()
            break
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice! Please enter 1, 2, or 3.")

def main():
    parser = argparse.ArgumentParser(description='Lung Cancer Detection System Launcher')
    parser.add_argument('--gui', action='store_true', help='Launch GUI directly')
    parser.add_argument('--cli', action='store_true', help='Launch CLI directly')
    
    args = parser.parse_args()
    
    if args.gui:
        # Check requirements and model before launching GUI
        if check_requirements():
            model_exists, _ = check_model()
            if model_exists:
                launch_gui()
            else:
                print("❌ No trained model found. Please train the model first.")
        
    elif args.cli:
        # Check requirements and model before launching CLI
        if check_requirements():
            model_exists, _ = check_model()
            if model_exists:
                launch_cli()
            else:
                print("❌ No trained model found. Please train the model first.")
    
    else:
        # Interactive mode
        interactive_launcher()

if __name__ == "__main__":
    main()