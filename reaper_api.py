%%writefile reaper_api.py
import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

dit_handler = AceStepHandler()
llm_handler = LLMHandler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    dit_handler.initialize_service(
        project_root="/content/ACE-Step-1.5", 
        config_path="acestep-v15-turbo",
        device="cuda"
    )
    llm_handler.initialize(
        checkpoint_dir="/content/checkpoints", 
        lm_model_path="acestep-5Hz-lm-0.6B",
        backend="vllm",
        device="cuda"
    )
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(
    caption: str = Form(...),
    bpm: int = Form(None),
    duration: float = Form(-1.0),
    task_type: str = Form("cover"),
    audio_file: UploadFile = File(None)
):
    src_audio_path = None

    if audio_file:
        temp_dir = "/tmp/acestep_inputs"
        os.makedirs(temp_dir, exist_ok=True)
        src_audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{audio_file.filename}")
        with open(src_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

    params = GenerationParams(
        task_type=task_type,
        caption=caption,
        src_audio=src_audio_path,
        bpm=bpm,
        duration=duration,
        shift=3.0,
        audio_cover_strength=1.0
    )

    config = GenerationConfig(
        batch_size=1,
        audio_format="wav"
    )

    save_directory = "/tmp/acestep_outputs"
    os.makedirs(save_directory, exist_ok=True)
    
    result = generate_music(
        dit_handler, 
        llm_handler, 
        params, 
        config, 
        save_dir=save_directory
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    generated_audio_path = result.audios[0]['path']
    
    return FileResponse(
        path=generated_audio_path,
        media_type="audio/wav",
        filename=os.path.basename(generated_audio_path)
    )
