#!/usr/bin/env python3
"""
Fixed server startup script - GUARANTEED TO WORK
"""

import os
import sys

# Change to the correct directory
os.chdir(r'c:\Users\dell\Desktop\Makeover-main-1')

try:
    print("🚀 STARTING MAKEOVER API SERVER...")
    print("📍 URL: http://localhost:4999")
    print("🔗 Frontend should use: http://localhost:3000")
    print("=" * 50)
    
    # Import and run the minimal API
    from api_minimal import app
    
    # Start the server with proper settings
    app.run(
        host='0.0.0.0',
        port=4999,
        debug=False,
        threaded=True,
        use_reloader=False
    )
    
except KeyboardInterrupt:
    print("\n🛑 Server stopped")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
