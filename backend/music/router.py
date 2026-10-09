import asyncio

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .services.composer import composer # import instance of Composer
from .services.composer import ComposerStateError, ModelNames

router = APIRouter()

# Serializes access to the shared `composer` singleton. The blocking calls below now
# run in a thread pool (so they no longer freeze the server), but that means multiple
# requests can run concurrently against the SAME composer instance (two tabs, a
# double-click, a retry). This lock prevents those from interleaving and corrupting
# composer's state. It does not fix the "one global session for everyone" limitation
# -- that needs the session-pool refactor -- it just makes the current singleton safe
# under concurrent requests.
composer_lock = asyncio.Lock()

class ModelSelection(BaseModel):
    model_name: str

class GenerateRequest(BaseModel):
    midi_id: int

@router.post("/midi/upload")
async def upload_midi(midiFile: UploadFile = File(...)):
    print("Received file:", midiFile.filename)
    content = await midiFile.read()
    async with composer_lock:
        await run_in_threadpool(composer.handleUpload, content) # fine tune and save model
    return {"status": "ok"}

@router.get("/model/selection") # sends all possible models to frontend
async def models():
    return {"models": ModelNames.allModels()}

@router.put("/model/select")
async def select_model(selection: ModelSelection):
    try:
        async with composer_lock:
            await run_in_threadpool(composer.selectModel, ModelNames(selection.model_name)) # also calls fineTune
    except ComposerStateError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": "Model selected successfully"}

@router.post("/generate")
async def generate_music(): # todo: frontend: load indicator while generating
    if composer.currentModelName == "Not Selected":
        return {"status": "Model Not Selected"}
    try:
        async with composer_lock:
            midi_bytes = await run_in_threadpool(composer.generateMusic)
    except ComposerStateError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(
        content=midi_bytes,
        media_type="audio/midi",
        headers={"Content-Disposition": "attachment; filename=generated.mid"},
    )
