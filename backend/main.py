from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router as api_router
from backend.config import settings

app = FastAPI(
    title="ToS Red-Flag Scanner & Clause Relationship Analyzer",
    description=(
        "An informational document-analysis backend for Terms of Service agreements. "
        "Not legal advice. Not a legal compliance or enforceability detector."
    ),
    version="0.1.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(api_router)

from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Mount frontend web interface
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")



@app.get("/")
async def root():
    """Root endpoint providing service metadata and disclaimer."""
    return {
        "service": "ToS Red-Flag Scanner API",
        "version": "0.1.0",
        "status": "online",
        "health_check": "/api/health",
        "web_interface": "/app/",
        "disclaimer": (
            "This tool is an informational document-analysis system, not a legal-advice service. "
            "It does not provide legal advice, compliance certification, or enforceability determinations."
        ),
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
