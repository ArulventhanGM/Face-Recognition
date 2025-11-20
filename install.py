#!/usr/bin/env python3
"""
Installation script for Face Recognition System
Handles dependency installation with fallbacks
"""

import subprocess
import sys
import os
import importlib
from typing import List, Tuple

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected.")
    return True

def install_package(package: str, fallback: str = None) -> bool:
    """Install a package with optional fallback"""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        if fallback:
            try:
                print(f"🔄 Trying fallback: {fallback}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", fallback])
                print(f"✅ {fallback} installed successfully")
                return True
            except subprocess.CalledProcessError:
                print(f"❌ Fallback {fallback} also failed")
        return False

def check_package(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is available")
        return True
    except ImportError:
        print(f"❌ {package_name} is not available")
        return False

def main():
    """Main installation process"""
    print("🚀 Face Recognition System Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Update pip first
    print("\n📦 Updating pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip updated successfully")
    except subprocess.CalledProcessError:
        print("⚠️  pip update failed, continuing...")
    
    # Core packages that should always work
    core_packages = [
        "numpy>=1.24.0",
        "opencv-python>=4.8.0", 
        "Pillow>=10.0.0",
        "Flask>=2.3.0",
        "Flask-Login>=0.6.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0"
    ]
    
    print("\n📦 Installing core packages...")
    failed_core = []
    for package in core_packages:
        if not install_package(package):
            failed_core.append(package)
    
    if failed_core:
        print(f"\n⚠️  Some core packages failed: {failed_core}")
        print("The system may still work with reduced functionality.")
    
    # Face recognition packages
    print("\n🔍 Installing face recognition packages...")
    
    # Try face-recognition with dlib fallback
    face_rec_installed = install_package("face-recognition>=1.3.0")
    if not face_rec_installed:
        print("⚠️  face-recognition failed. Trying individual components...")
        install_package("dlib>=19.24.0")
        install_package("face-recognition-models>=0.3.0")
    
    # Optional advanced packages
    print("\n🧠 Installing optional advanced packages...")
    
    advanced_packages = [
        ("torch>=2.0.0", "torch"),
        ("torchvision>=0.15.0", "torchvision"), 
        ("facenet-pytorch>=2.5.3", None)
    ]
    
    for package, fallback in advanced_packages:
        success = install_package(package, fallback)
        if not success:
            print(f"⚠️  {package} installation failed. Advanced features may be limited.")
    
    # Check final installation status
    print("\n🔍 Checking installation status...")
    
    check_packages = [
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("Flask", "flask"),
        ("scikit-learn", "sklearn"),
        ("face-recognition", "face_recognition"),
        ("torch", "torch"),
        ("PIL", "PIL")
    ]
    
    available_count = 0
    for package_name, import_name in check_packages:
        if check_package(package_name, import_name):
            available_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ {available_count}/{len(check_packages)} packages available")
    
    if available_count >= 6:  # Core packages work
        print("🎉 Installation successful! The system should work.")
        print("\n🚀 You can now run:")
        print("   python app.py")
    elif available_count >= 4:
        print("⚠️  Partial installation. Basic functionality should work.")
        print("   Some advanced features may be disabled.")
    else:
        print("❌ Installation incomplete. Please check error messages above.")
        return 1
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    print("\n📁 Created models directory")
    
    print("\n✨ Setup complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
