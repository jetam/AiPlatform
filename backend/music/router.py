from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .services.composer import ComposerStateError, ModelNames
from .services.session_pool import SESSION_COOKIE_NAME, new_session_id, session_pool

router = APIRouter()


async def get_session(request: Request) -> tuple[str, dict]:

    session_id = request.cookies.get(SESSION_COOKIE_NAME) or new_session_id()
    entry = await session_pool.get(session_id)
    return session_id, entry


def _set_session_cookie(response: Response, session_id: str):
    response.set_cookie(SESSION_COOKIE_NAME, session_id, httponly=True, samesite="lax")


class ModelSelection(BaseModel):
    model_name: str

class GenerateRequest(BaseModel):
    midi_id: int

@router.post("/midi/upload")
async def upload_midi( response: Response, midiFile: UploadFile = File(...), session: tuple = Depends(get_session)):
    session_id, entry = session
    print("Received file:", midiFile.filename)
    content = await midiFile.read()
    async with entry["lock"]:
        await run_in_threadpool(entry["composer"].handleUpload, content) # fine tune and save model
    _set_session_cookie(response, session_id)
    return {"status": "ok"}

@router.get("/model/selection")
async def models():
    return {"models": ModelNames.allModels()}

@router.put("/model/select")
async def select_model( response: Response, selection: ModelSelection, session: tuple = Depends(get_session)):
    session_id, entry = session
    try:
        async with entry["lock"]:
            await run_in_threadpool(entry["composer"].selectModel, ModelNames(selection.model_name)) # also calls fineTune
    except ComposerStateError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _set_session_cookie(response, session_id)
    return {"message": "Model selected successfully"}

@router.post("/generate")
async def generate_music( response: Response, session: tuple = Depends(get_session)): # todo: frontend: load indicator while generating
    session_id, entry = session
    composer = entry["composer"]

    if composer.currentModelName == "Not Selected":
        _set_session_cookie(response, session_id)
        return {"status": "Model Not Selected"}

    try:
        async with entry["lock"]:
            midi_bytes = await run_in_threadpool(composer.generateMusic)
    except ComposerStateError as e:
        raise HTTPException(status_code=400, detail=str(e))

    final_response = Response(
        content=midi_bytes,
        media_type="audio/midi",
        headers={"Content-Disposition": "attachment; filename=generated.mid"},
    )
    _set_session_cookie(final_response, session_id)
    return final_response
