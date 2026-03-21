import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .chunker import chunk_pdf_files, DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP
from .embedder import DocumentEmbedder

try:
    from ..tableHandling.tablePdfRead import extract_tables_from_pdf
    from ..tableHandling.trialNLSentence import enrich_tables_with_sentences
except ImportError:
    from src.tableHandling.tablePdfRead import extract_tables_from_pdf
    from src.tableHandling.trialNLSentence import enrich_tables_with_sentences


class IngestionPipeline:
    def __init__(
        self,
        dataset_dir: str,
        collection_name: str = "temp_dataset",
        chunk_output_dir: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        table_sentence_model: str = "mistral-medium-latest",
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            dataset_dir: Directory containing PDF files
            collection_name: Name for the ChromaDB collection
            chunk_output_dir: Optional directory to save JSON outputs
            file_paths: Optional list of PDF file paths to ingest
            table_sentence_model: Mistral model used for table sentence generation
        """
        self.dataset_dir = Path(dataset_dir)
        self.chunk_output_dir = Path(chunk_output_dir) if chunk_output_dir else None
        self.collection_name = collection_name
        self.file_paths = file_paths
        self.table_sentence_model = table_sentence_model

        self.embedder = DocumentEmbedder(collection_name=collection_name)
        self.document_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.document_tables: Dict[str, List[Dict[str, Any]]] = {}
        self.document_table_sentences: Dict[str, List[Dict[str, Any]]] = {}
        self.document_table_sentence_records: Dict[str, List[Dict[str, Any]]] = {}

    def _validated_file_paths(self) -> List[str]:
        valid_paths: List[str] = []
        for file_path in self.file_paths or []:
            path = Path(file_path)
            if not path.exists():
                print(f"Skipping missing file: {file_path}")
                continue
            if path.suffix.lower() != ".pdf":
                print(f"Skipping non-PDF file: {file_path}")
                continue
            valid_paths.append(str(path))
        return valid_paths

    def _resolve_input_files(self) -> List[str]:
        if self.file_paths:
            return self._validated_file_paths()

        return [
            str(path)
            for path in sorted(self.dataset_dir.iterdir())
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]

    def _prepare_output_dirs(self) -> None:
        if not self.chunk_output_dir:
            return

        self.chunk_output_dir.mkdir(parents=True, exist_ok=True)
        (self.chunk_output_dir / "chunks").mkdir(parents=True, exist_ok=True)
        (self.chunk_output_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.chunk_output_dir / "tables_nl").mkdir(parents=True, exist_ok=True)

    def _json_output_path(self, category: str, file_name: str) -> Optional[Path]:
        if not self.chunk_output_dir:
            return None

        file_stem = Path(file_name).stem
        if category == "chunks":
            return self.chunk_output_dir / "chunks" / f"{file_stem}_chunks.json"
        if category == "tables":
            return self.chunk_output_dir / "tables" / f"{file_stem}_tables.json"
        if category == "tables_nl":
            return self.chunk_output_dir / "tables_nl" / f"{file_stem}_tables_nl.json"
        return None

    def _save_json(self, path: Optional[Path], data: Any) -> None:
        if path is None:
            return
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _build_table_sentence_records(
        self,
        file_name: str,
        enriched_tables: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []

        for table_position, table in enumerate(enriched_tables, start=1):
            heading = str(table.get("heading", "") or "").strip()
            section_heading = heading or "General"
            page_number = int(table.get("page_number") or 0)
            end_page = int(table.get("end_page") or page_number)
            table_index = int(table.get("table_index") or table_position)
            sentences = table.get("nl_sentences") or []

            for sentence_index, sentence in enumerate(sentences, start=1):
                text = str(sentence).strip()
                if not text:
                    continue

                records.append(
                    {
                        "chunk_id": f"{file_name}_table_{table_position}_sentence_{sentence_index}",
                        "text": text,
                        "metadata": {
                            "document_name": file_name,
                            "section_heading": section_heading,
                            "clause_id": "",
                            "content_type": "table_sentence",
                            "table_heading": heading,
                            "table_index": table_index,
                            "page_number": page_number,
                            "end_page": end_page,
                            "row_index": sentence_index,
                            "spans_pages": bool(table.get("spans_pages", False)),
                            "nl_status": str(table.get("nl_status", "") or ""),
                        },
                    }
                )

        return records

    def _process_document_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        file_name = Path(file_path).name
        print(f"[{file_name}] Thread 1 -> chunking + embedding prose")
        chunk_output_file = self._json_output_path("chunks", file_name)
        chunks = chunk_pdf_files(
            file_paths=[file_path],
            output_file=str(chunk_output_file) if chunk_output_file else None,
        )
        self.embedder.embed_documents(chunks)
        return chunks

    def _process_document_tables(
        self,
        file_path: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        file_name = Path(file_path).name
        print(f"[{file_name}] Thread 2 -> extracting tables")
        tables = extract_tables_from_pdf(file_path)
        self._save_json(self._json_output_path("tables", file_name), tables)

        print(f"[{file_name}] Thread 2 -> generating NL sentences for tables")
        enriched_tables = enrich_tables_with_sentences(
            tables=tables,
            model=self.table_sentence_model,
        )
        self._save_json(self._json_output_path("tables_nl", file_name), enriched_tables)

        sentence_records = self._build_table_sentence_records(file_name, enriched_tables)
        if sentence_records:
            print(f"[{file_name}] Thread 2 -> embedding {len(sentence_records)} table sentences")
            self.embedder.embed_records(sentence_records)
        else:
            print(f"[{file_name}] Thread 2 -> no table sentences to embed")

        return tables, enriched_tables, sentence_records

    def _process_single_document(self, file_path: str) -> None:
        file_name = Path(file_path).name
        print(f"Processing document: {file_name}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            chunk_future = executor.submit(self._process_document_chunks, file_path)
            table_future = executor.submit(self._process_document_tables, file_path)

            chunks = chunk_future.result()
            tables, enriched_tables, sentence_records = table_future.result()

        self.document_chunks[file_name] = chunks
        self.document_tables[file_name] = tables
        self.document_table_sentences[file_name] = enriched_tables
        self.document_table_sentence_records[file_name] = sentence_records

        print(
            f"Completed {file_name} -> prose_chunks={len(chunks)}, "
            f"tables={len(tables)}, table_sentences={len(sentence_records)}"
        )

    def _save_aggregate_outputs(self) -> None:
        if not self.chunk_output_dir:
            return

        all_chunks = [chunk for chunks in self.document_chunks.values() for chunk in chunks]
        all_tables = [table for tables in self.document_tables.values() for table in tables]
        all_table_sentences = [
            table for enriched_tables in self.document_table_sentences.values() for table in enriched_tables
        ]

        self._save_json(self.chunk_output_dir / "chunks.json", all_chunks)
        self._save_json(self.chunk_output_dir / "tables.json", all_tables)
        self._save_json(self.chunk_output_dir / "tables_nl.json", all_table_sentences)

    def run(self) -> None:
        """
        Run the ingestion pipeline per document:
        1. Thread 1: chunk text and embed chunks
        2. Thread 2: extract tables, generate NL sentences, and embed table sentences
        """
        input_files = self._resolve_input_files()
        if self.file_paths:
            print(f"Processing PDFs: {len(input_files)} file(s)")
        else:
            print(f"Processing PDFs from: {self.dataset_dir}")
        print(
            f"Chunking config - max_tokens: {DEFAULT_MAX_TOKENS}, overlap: {int(DEFAULT_OVERLAP * 100)}%"
        )

        if not input_files:
            print("No valid PDF files to process.")
            return

        self.document_chunks.clear()
        self.document_tables.clear()
        self.document_table_sentences.clear()
        self.document_table_sentence_records.clear()
        self._prepare_output_dirs()

        try:
            for file_path in input_files:
                self._process_single_document(file_path)

            self._save_aggregate_outputs()

            total_chunks = sum(len(chunks) for chunks in self.document_chunks.values())
            total_tables = sum(len(tables) for tables in self.document_tables.values())
            total_table_sentences = sum(
                len(records) for records in self.document_table_sentence_records.values()
            )
            print(
                "Ingestion pipeline completed successfully. "
                f"Prose chunks: {total_chunks}, tables: {total_tables}, "
                f"table sentences: {total_table_sentences}"
            )
        except Exception as e:
            print(f"Error during ingestion: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    dataset_dir = r"D:\_official_\_MIT ADT_\_SEMESTER 7_\MP5\MP5-RAG-System\src\ingestion\temp_dataset"
    pipeline = IngestionPipeline(
        dataset_dir=dataset_dir,
        collection_name="temp_dataset",
        chunk_output_dir=None,
    )
    pipeline.run()
