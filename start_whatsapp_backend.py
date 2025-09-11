#!/usr/bin/env python3
"""
WhatsApp Backend Startup Script
Production-ready startup for the modular WhatsApp backend system
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def configure_logging(debug_mode: bool):
    """Configure logging to reduce noise from WebSocket ping/pong messages"""
    
    if debug_mode:
        # Set our application loggers to DEBUG
        logging.getLogger("whatsapp_backend").setLevel(logging.DEBUG)
        logging.getLogger("rag_system").setLevel(logging.DEBUG)
        logging.getLogger("bitrix_integration").setLevel(logging.DEBUG)
        
        # Reduce noise from uvicorn WebSocket messages
        logging.getLogger("uvicorn.protocols.websockets").setLevel(logging.WARNING)
        logging.getLogger("websockets.protocol").setLevel(logging.WARNING)
        logging.getLogger("websockets.server").setLevel(logging.WARNING)
        
        # Keep uvicorn main logs at INFO (connection open/close)
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
        
    else:
        # Production mode - keep it clean
        logging.getLogger("whatsapp_backend").setLevel(logging.INFO)
        logging.getLogger("rag_system").setLevel(logging.INFO)
        logging.getLogger("bitrix_integration").setLevel(logging.INFO)
        logging.getLogger("uvicorn").setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Start WhatsApp Backend System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--mode", choices=["test", "production"], default="test", help="Operating mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["WHATSAPP_TEST_MODE"] = "true" if args.mode == "test" else "false"
    os.environ["DEBUG"] = "true" if args.debug else "false"
    
    try:
        import uvicorn
        
        print("🚀 Starting WhatsApp Backend System")
        print("=" * 50)
        print(f"📱 Mode: {args.mode.upper()}")
        print(f"🔗 Server: http://{args.host}:{args.port}")
        print(f"📊 Status: http://{args.host}:{args.port}/status")
        print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
        print(f"🧪 WebSocket: ws://{args.host}:{args.port}/ws/{{phone}}")
        if args.mode == "test":
            print(f"🌐 Test UI: http://{args.host}:{args.port}")
        print("=" * 50)
        
        # Configure custom logging to reduce WebSocket noise
        configure_logging(args.debug)
        
        # Use import string for reload mode, app object for production
        if args.reload or args.debug:
            app_str = "whatsapp_backend.app:app"
            uvicorn.run(
                app_str,
                host=args.host,
                port=args.port,
                reload=True,
                log_level="info"  # Use INFO level, our custom config handles DEBUG for app loggers
            )
        else:
            from whatsapp_backend.app import app
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                reload=False,
                log_level="info"  # Use INFO level, our custom config handles DEBUG for app loggers
            )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install dependencies: pip install uvicorn fastapi")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
