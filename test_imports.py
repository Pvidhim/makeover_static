#!/usr/bin/env python3
"""
Test script to check if all imports work correctly
"""

print("🔄 Testing imports...")

try:
    print("  - Testing Flask...")
    from flask import Flask, request, jsonify
    print("    ✅ Flask OK")
except Exception as e:
    print(f"    ❌ Flask failed: {e}")

try:
    print("  - Testing PIL...")
    from PIL import Image
    print("    ✅ PIL OK")
except Exception as e:
    print(f"    ❌ PIL failed: {e}")

try:
    print("  - Testing OpenCV...")
    import cv2
    print(f"    ✅ OpenCV OK (version: {cv2.__version__})")
except Exception as e:
    print(f"    ❌ OpenCV failed: {e}")

try:
    print("  - Testing dlib...")
    import dlib
    print("    ✅ dlib OK")
except Exception as e:
    print(f"    ❌ dlib failed: {e}")

try:
    print("  - Testing MediaPipe...")
    import mediapipe as mp
    print("    ✅ MediaPipe OK")
except Exception as e:
    print(f"    ❌ MediaPipe failed: {e}")

try:
    print("  - Testing torch...")
    import torch
    print(f"    ✅ PyTorch OK (version: {torch.__version__})")
    
    # Test basic tensor operations
    print("  - Testing torch tensor...")
    x = torch.tensor([1, 2, 3])
    print(f"    ✅ Tensor creation OK: {x}")
    
except Exception as e:
    print(f"    ❌ PyTorch failed: {e}")

try:
    print("  - Testing numpy...")
    import numpy as np
    print(f"    ✅ NumPy OK (version: {np.__version__})")
except Exception as e:
    print(f"    ❌ NumPy failed: {e}")

print("🎉 Import test completed!")
