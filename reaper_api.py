import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn

app = FastAPI(title="ACE-Step REAPER Bridge")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Servidor ACE-Step REAPER Bridge rodando!"}

@app.post("/process_audio/")
async def process_audio(
    file: UploadFile = File(...), 
    prompt: str = Form(""), 
    bpm: float = Form(120.0)
):
    # 1. Salva o áudio recebido do REAPER temporariamente
    temp_in = f"temp_input_{file.filename}"
    with open(temp_in, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # TODO: Aqui chamaremos o pipeline do ACE-Step para processar o inpainting
    # output_path = ace_step_inpainting(temp_in, prompt, bpm)
    
    # Por enquanto, devolvemos o próprio arquivo recebido para testar a ponte
    return {"status": "success", "prompt": prompt, "bpm": bpm, "file_received": file.filename}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
