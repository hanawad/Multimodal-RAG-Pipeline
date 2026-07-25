from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from backend.config import UPLOAD_DIR
from backend.rag_service import (
    ask_with_rag,
    ask_without_rag,
    ingest_csv,
    ingest_docx,
    ingest_image,
    ingest_pdf,
)
from backend.schemas import AskRequest, AskResponse
from backend.utils import ensure_directories, save_uploaded_file


@asynccontextmanager
async def application_lifespan(app_instance: FastAPI):
    """Lifecycle manager to execute startup tasks before server accepts requests."""
    ensure_directories()
    yield


app = FastAPI(
    title="Multimodal RAG Teaching Engine",
    description="Interactive API for PDF and Document Retrieval-Augmented Generation.",
    version="2.0.0",
    lifespan=application_lifespan,
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def process_file_upload(file: UploadFile = File(...)):
    """Handles file ingestion across multiple document formats (PDF, DOCX, CSV, Image)."""
    saved_path = save_uploaded_file(file, UPLOAD_DIR)
    file_extension = Path(file.filename).suffix.lower()

    match file_extension:
        case ".pdf":
            indexing_details = ingest_pdf(saved_path)
            detected_type = "pdf"
        case ".png" | ".jpg" | ".jpeg":
            indexing_details = ingest_image(saved_path)
            detected_type = "image"
        case ".csv":
            indexing_details = ingest_csv(saved_path)
            detected_type = "csv"
        case ".docx":
            indexing_details = ingest_docx(saved_path)
            detected_type = "docx"
        case _:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format '{file_extension}'. Supported: .pdf, .docx, .csv, .png, .jpg",
            )

    return {
        "filename": file.filename,
        "file_type": detected_type,
        **indexing_details,
    }


@app.post("/ask", response_model=AskResponse)
async def query_pipeline(request: AskRequest):
    """Generates dual responses comparing direct LLM output versus grounded RAG output."""
    baseline_response = ask_without_rag(request.question)
    rag_response = ask_with_rag(request.question)

    return AskResponse(
        question=request.question,
        no_rag_answer=baseline_response,
        rag_answer=rag_response["answer"],
        retrieved_chunks=rag_response["chunks"],
    )


@app.get("/health")
async def check_health():
    """Health check endpoint to verify backend operational status."""
    return {"status": "ok", "service": "Multimodal RAG Service"}