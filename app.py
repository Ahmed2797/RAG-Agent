from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager
import os, shutil, time

# ✅ Correct imports
from ragagent.pipeline import pipeline, PDF_CREW_CACHE, get_pdf_crew

APP_NAME = "PDF Chat Assistant"
UPLOAD_DIR = "uploads"
INDEX_FILE = "ragagent/index.html"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------
# GLOBAL STATE
# ----------------------
active_pdf_path: Optional[str] = None

# ----------------------
# MODELS
# ----------------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    processing_time: float
    success: bool

# ----------------------
# UTILITIES
# ----------------------
def list_pdfs() -> List[dict]:
    pdfs = []
    for f in os.listdir(UPLOAD_DIR):
        if f.lower().endswith(".pdf"):
            path = os.path.join(UPLOAD_DIR, f)
            stat = os.stat(path)
            pdfs.append({
                "name": f,
                "size": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "is_active": path == active_pdf_path
            })
    return pdfs

def set_active_pdf_from_folder():
    """Set first PDF as active on startup"""
    global active_pdf_path
    pdfs = list_pdfs()
    if pdfs:
        active_pdf_path = os.path.join(UPLOAD_DIR, pdfs[0]["name"])
        get_pdf_crew(active_pdf_path)   # cache crew
        print(f"[INFO] Active PDF set: {active_pdf_path}")
    else:
        active_pdf_path = None
        print("[INFO] No PDFs found.")

# ----------------------
# LIFESPAN
# ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Starting up...")
    set_active_pdf_from_folder()
    yield
    print("[INFO] Shutting down...")

# ----------------------
# FASTAPI APP
# ----------------------
app = FastAPI(title=APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# FRONTEND
# ----------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    if not os.path.exists(INDEX_FILE):
        return HTMLResponse("<h1>index.html not found</h1>")
    return open(INDEX_FILE, encoding="utf-8").read()

# ----------------------
# UPLOAD PDF
# ----------------------
@app.post("/api/upload")
def upload_pdf(file: UploadFile = File(...)):
    global active_pdf_path

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed")

    filename = f"{int(time.time())}_{file.filename.replace(' ', '_')}"
    path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(500, f"Failed to save PDF: {e}")

    # set active
    active_pdf_path = path

    # cache crew (embedding happens automatically on first query)
    get_pdf_crew(path)

    print(f"[INFO] Uploaded PDF and cached crew: {active_pdf_path}")
    return {"success": True, "active_pdf": filename}

# ----------------------
# DELETE PDF
# ----------------------
@app.delete("/api/delete/{pdf_name}")
def delete_pdf(pdf_name: str):
    global active_pdf_path

    file_path = os.path.join(UPLOAD_DIR, pdf_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    os.remove(file_path)

    # remove from cache
    PDF_CREW_CACHE.pop(file_path, None)

    # reset active if needed
    if active_pdf_path == file_path:
        pdfs = list_pdfs()
        if pdfs:
            active_pdf_path = os.path.join(UPLOAD_DIR, pdfs[0]["name"])
            get_pdf_crew(active_pdf_path)
        else:
            active_pdf_path = None

    return {
        "success": True,
        "message": f"{pdf_name} deleted",
        "active_pdf": active_pdf_path
    }

# ----------------------
# SWITCH PDF
# ----------------------
@app.post("/api/switch/{pdf_name}")
def switch_pdf(pdf_name: str):
    global active_pdf_path

    file_path = os.path.join(UPLOAD_DIR, pdf_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    active_pdf_path = file_path
    get_pdf_crew(active_pdf_path)

    return {"success": True, "active_pdf": pdf_name}

# ----------------------
# LIST PDFS
# ----------------------
@app.get("/api/pdfs")
def get_pdfs():
    return list_pdfs()

@app.get("/api/active-pdf")
def get_active_pdf():
    return {"active_pdf": active_pdf_path}

# ----------------------
# CHAT
# ----------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        return ChatResponse(answer="Empty question", processing_time=0, success=False)

    if not active_pdf_path or not os.path.exists(active_pdf_path):
        return ChatResponse(
            answer="No active PDF. Upload one first.",
            processing_time=0,
            success=False
        )

    start = time.time()
    answer = pipeline(req.message, active_pdf_path)
    duration = round(time.time() - start, 3)

    return ChatResponse(
        answer=answer,
        processing_time=duration,
        success=True
    )

# ----------------------
# HEALTH
# ----------------------
@app.get("/api/health")
def health():
    return {"status": "healthy", "active_pdf": active_pdf_path}

# ----------------------
# RUN
# ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)


### uvicorn app:app --reload
