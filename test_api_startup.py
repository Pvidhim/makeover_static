#!/usr/bin/env python3
"""
Test if the API can start without errors
"""

import sys
import os

print("🔧 Testing API startup...")

try:
    # Change to the correct directory
    os.chdir(r'c:\Users\dell\Desktop\Makeover-main-1')
    
    # Import the API (this will test if all imports work)
    print("  - Testing imports...")
    import api
    
    print("  ✅ API imports successful!")
    print("  - Models loaded:", api.models_loaded)
    print("  - Flask app created:", type(api.app))
    
    print("🎉 API startup test successful!")
    
except Exception as e:
    print(f"❌ API startup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
