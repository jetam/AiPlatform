
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from music import router as musicRouter

app = FastAPI(title="AI Music + Market Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("start server")

app.include_router(musicRouter.router, prefix="/api/music")

@app.get("/health")
def health_check():
    return {"status": "ok"}
