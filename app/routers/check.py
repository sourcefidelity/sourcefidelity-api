"""Submit a paper for citation checking."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

router = APIRouter()


@router.post("/")
async def check_paper(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    store_only: bool = Form(False),
):
    """
    Submit a paper for citation checking.

    - **file**: PDF or DOCX file
    - **title**: Optional paper title (auto-detected if not provided)
    - **store_only**: If True, only extract and store text (no verification)
    """
    # Validate file type
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PDF or DOCX.",
        )

    # Placeholder – will be wired to Celery task in Phase 5
    return {
        "message": "Endpoint not yet implemented (Phase 4)",
        "filename": file.filename,
        "content_type": file.content_type,
    }
