#!/usr/bin/env python3
"""
Minimal Flask test to check basic server startup
"""

print("🔄 Starting minimal Flask test...")

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "Basic Flask is working!"})

if __name__ == '__main__':
    print("✅ Flask app created successfully")
    print("🚀 Starting server on port 4999...")
    try:
        app.run(debug=True, host='0.0.0.0', port=4999, threaded=True)
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
