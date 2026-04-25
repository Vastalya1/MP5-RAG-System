from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "combined_evaluation_dataset_hybrid_chatgpt.json"
    default_output = default_input.with_suffix(".xlsx")

    parser = argparse.ArgumentParser(
        description="Convert a combined evaluation dataset JSON file into an Excel workbook."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Path to the JSON file. Default: {default_input}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Path to the Excel file to create. Default: {default_output}",
    )
    parser.add_argument(
        "--sheet-name",
        default="evaluation_dataset",
        help="Worksheet name for the exported data.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("Expected the JSON file to contain a list of objects.")

    return data


def format_cell_value(column_name: str, value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            return ""

        separator = "\n\n" if "contexts" in column_name else "\n"
        if all(not isinstance(item, (dict, list)) for item in value):
            return separator.join("" if item is None else str(item) for item in value)

        return json.dumps(value, ensure_ascii=False, indent=2)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)

    return value


def build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    column_order = list(records[0].keys())
    processed_records = []

    for record in records:
        processed_record = {
            key: format_cell_value(key, value) for key, value in record.items()
        }
        processed_records.append(processed_record)

        for key in processed_record:
            if key not in column_order:
                column_order.append(key)

    return pd.DataFrame(processed_records, columns=column_order)


def format_worksheet(worksheet) -> None:
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="top", wrap_text=True)

    for column_cells in worksheet.columns:
        column_index = column_cells[0].column
        column_letter = get_column_letter(column_index)
        max_length = 0

        for cell in column_cells:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            value = "" if cell.value is None else str(cell.value)
            longest_line = max((len(line) for line in value.splitlines()), default=0)
            max_length = max(max_length, longest_line)

        worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 14), 60)


def export_to_excel(dataframe: pd.DataFrame, output_path: Path, sheet_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.book[sheet_name]
        format_worksheet(worksheet)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.input.resolve())
    dataframe = build_dataframe(dataset)
    export_to_excel(dataframe, args.output.resolve(), args.sheet_name)
    print(f"Excel file created: {args.output.resolve()}")


if __name__ == "__main__":
    main()
