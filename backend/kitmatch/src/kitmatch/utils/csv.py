import csv
import io
from typing import List, Dict, Any


def to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    output = io.StringIO()
    fieldnames = ["image", "class", "confidence", "x_min", "y_min", "x_max", "y_max"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return output.getvalue().encode("utf-8")


