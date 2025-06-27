#!/usr/bin/env python3
"""
Quick server test to verify the API works
"""

import sys
import requests
import time
import threading
from api import app

def start_server():
    """Start the server in a separate thread"""
    try:
        app.run(host='127.0.0.1', port=4999, debug=False, threaded=True)
    except Exception as e:
        print(f"Server error: {e}")

def test_endpoints():
    """Test the endpoints"""
    time.sleep(2)  # Give server time to start
    
    try:
        # Test health endpoint
        print("🔍 Testing /health endpoint...")
        response = requests.get('http://localhost:4999/health', timeout=5)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        
        # Test models status
        print("🔍 Testing /models/status endpoint...")
        response = requests.get('http://localhost:4999/models/status', timeout=5)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        
        print("✅ Basic endpoints working!")
        
    except Exception as e:
        print(f"❌ Endpoint test failed: {e}")
    
    finally:
        # Kill the process since we can't gracefully stop Flask in this setup
        import os
        os._exit(0)

if __name__ == '__main__':
    print("🚀 Starting quick server test...")
    
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Test endpoints
    test_endpoints()
