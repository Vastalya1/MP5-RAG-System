from pathlib import Path
from typing import Iterable, List, Optional
import shutil

from PIL import Image

try:
    from docx import Document  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Document = None  # type: ignore

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


PDF_EXTENSION = ".pdf"
SUPPORTED_DOC_EXTENSIONS = {".docx"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _ensure_dir(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)


def _safe_output_path(input_file: Path, dataset_root: Path, output_root: Path, new_ext: Optional[str] = None) -> Path:
    relative = input_file.relative_to(dataset_root)
    if new_ext is not None:
        relative = relative.with_suffix(new_ext)
    out_path = output_root / relative
    _ensure_dir(out_path.parent)
    return out_path


def _docx_to_text(path: Path) -> str:
    if Document is None:
        raise RuntimeError(
            "python-docx is required for .docx conversion. Install with 'pip install python-docx'."
        )
    doc = Document(str(path))
    paragraphs: List[str] = []
    for paragraph in doc.paragraphs:
        paragraphs.append(paragraph.text)
    return "\n".join(paragraphs)


def _write_text_pdf(text: str, output_pdf: Path, title: Optional[str] = None) -> None:
    doc = SimpleDocTemplate(str(output_pdf), pagesize=A4)
    styles = getSampleStyleSheet()
    story: List[object] = []
    if title:
        story.append(Paragraph(title, styles["Title"]))
        story.append(Spacer(1, 12))
    for line in text.split("\n"):
        story.append(Paragraph(line.replace("&", "&amp;"), styles["BodyText"]))
    doc.build(story)


def _convert_image_to_pdf(image_path: Path, output_pdf: Path) -> None:
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        rgb_img.save(output_pdf, save_all=True)


def convert_file_to_pdf(input_file: Path, dataset_root: Path, output_root: Path) -> Optional[Path]:
    suffix = input_file.suffix.lower()
    try:
        if suffix == PDF_EXTENSION:
            out_path = _safe_output_path(input_file, dataset_root, output_root)
            shutil.copy2(input_file, out_path)
            return out_path

        if suffix in SUPPORTED_DOC_EXTENSIONS:
            extracted = _docx_to_text(input_file)
            out_path = _safe_output_path(input_file, dataset_root, output_root, PDF_EXTENSION)
            _write_text_pdf(extracted, out_path, title=input_file.stem)
            return out_path

        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            out_path = _safe_output_path(input_file, dataset_root, output_root, PDF_EXTENSION)
            _convert_image_to_pdf(input_file, out_path)
            return out_path
    except Exception as exc:
        print(f"[WARN] Failed to convert {input_file}: {exc}")
        return None

    return None


def convert_dataset_to_pdfs(input_dir: str, output_dir: str) -> List[Path]:
    dataset_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {dataset_root}")

    converted: List[Path] = []
    for file_path in _iter_files(dataset_root):
        result = convert_file_to_pdf(file_path, dataset_root, output_root)
        if result is not None:
            converted.append(result)

    return converted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Copy PDFs and convert DOCX/images to PDF from an input directory.")
    parser.add_argument("--input", required=True, help="Path to the input directory")
    parser.add_argument("--output", required=True, help="Path to the output directory for PDFs")
    args = parser.parse_args()

    outputs = convert_dataset_to_pdfs(args.input, args.output)
    print(f"Converted {len(outputs)} file(s) to PDF.")

