"""
Main WhatsApp Backend Application
Modular, production-ready backend for WhatsApp chatbot integration
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import config
from .api.routes import router

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{config.LOGS_DIR}/whatsapp_backend.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting WhatsApp Backend System")
    logger.info("=" * 50)
    
    # Validate configuration
    is_valid, missing_keys = config.validate_required_keys()
    if not is_valid:
        logger.error(f"❌ Missing required configuration: {missing_keys}")
        logger.warning("⚠️ System will start but may not function properly")
    else:
        logger.info("✅ Configuration validated successfully")
    
    # Log system info
    config_status = config.get_status()
    logger.info(f"🎯 Mode: {config_status['mode']}")
    logger.info(f"📱 WhatsApp Line: {config_status['line_id']}")
    logger.info(f"🔗 Server: http://{config.HOST}:{config.PORT}")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down WhatsApp Backend System")

# Create FastAPI application
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="WhatsApp Backend System",
        description="Modular backend for WhatsApp chatbot integration with Leo & Loona",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    return app

# Create application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting WhatsApp Backend System")
    print("=" * 50)
    print(f"📱 Mode: {'TEST' if config.TEST_MODE else 'PRODUCTION'}")
    print(f"🔗 Server: http://{config.HOST}:{config.PORT}")
    print(f"📊 Status: http://{config.HOST}:{config.PORT}/status")
    print(f"📚 API Docs: http://{config.HOST}:{config.PORT}/docs")
    print(f"🧪 WebSocket: ws://{config.HOST}:{config.PORT}/ws/{{phone}}")
    print("=" * 50)
    
    uvicorn.run(
        "whatsapp_backend.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )
