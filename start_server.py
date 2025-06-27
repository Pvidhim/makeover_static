#!/usr/bin/env python3
"""
Manual server starter with better error handling
"""

import sys
import os

print("🚀 Starting Flask API Server manually...")

try:
    # Change to correct directory
    os.chdir(r'c:\Users\dell\Desktop\Makeover-main-1')
    
    # Import the api module
    from api import app, models_loaded
    
    print(f"✅ API imported successfully")
    print(f"📊 Models loaded: {models_loaded}")
    print(f"🌐 CORS configured for: localhost:3000")
    print(f"📍 Starting server on http://localhost:4999")
    print("💡 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start the server
    app.run(
        host='0.0.0.0',
        port=4999,
        debug=False,  # Turn off debug mode to avoid reloader issues
        threaded=True,
        use_reloader=False  # Disable reloader to prevent multiprocessing issues
    )
    
except KeyboardInterrupt:
    print("\n🛑 Server stopped by user")
    
except Exception as e:
    print(f"❌ Server startup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
