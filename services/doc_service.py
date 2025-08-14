from typing import List
import io
from PyPDF2 import PdfReader
import docx2txt

def _pdf_to_text(file) -> str:
    text = []
    reader = PdfReader(file)
    for page in reader.pages:
        txt = page.extract_text() or ""
        text.append(txt)
    return "\n".join(text)

def _docx_to_text(file) -> str:
    # file is UploadedFile; need bytes
    b = file.read()
    mem = io.BytesIO(b)
    return docx2txt.process(mem) or ""

def _txt_to_text(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def extract_text_from_files(files: List) -> str:
    chunks = []
    for f in files:
        name = f.name.lower()
        if name.endswith(".pdf"):
            chunks.append(_pdf_to_text(f))
        elif name.endswith(".docx"):
            chunks.append(_docx_to_text(f))
        elif name.endswith(".txt"):
            chunks.append(_txt_to_text(f))
    return "\n\n".join(chunks)
