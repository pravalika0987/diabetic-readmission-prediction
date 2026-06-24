import sqlite3
import math
from statistics import StatisticsError, mean, median, stdev

DB_PATH = "diabetic_db.sqlite"
REPORT_PATH = "data_quality_report.txt"
NUMERIC_TYPES = {"INTEGER", "REAL", "NUMERIC", "INT", "FLOAT", "DOUBLE", "DECIMAL"}


def get_tables(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [row[0] for row in cursor.fetchall()]


def get_table_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info('{table_name}')")
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2].upper().strip() if row[2] else "",
            "notnull": bool(row[3]),
            "default_value": row[4],
            "pk": bool(row[5]),
        }
        for row in cursor.fetchall()
    ]


def normalize_declared_type(declared_type):
    if not declared_type:
        return "TEXT"
    header = declared_type.split("(")[0].strip().upper()
    if header in NUMERIC_TYPES:
        return header
    if header.startswith("INT"):
        return "INTEGER"
    if header.startswith("CHAR") or header.startswith("CLOB") or header.startswith("TEXT"):
        return "TEXT"
    if header.startswith("BLOB"):
        return "BLOB"
    return declared_type


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detect_outliers(values):
    if len(values) < 4:
        return []
    sorted_values = sorted(values)
    q1 = median(sorted_values[: len(sorted_values) // 2])
    q3 = median(sorted_values[(len(sorted_values) + 1) // 2 :])
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    return [v for v in values if v < lower or v > upper]


def summarize_table(cursor, table_name):
    columns = get_table_columns(cursor, table_name)
    cursor.execute(f"SELECT * FROM '{table_name}'")
    rows = cursor.fetchall()
    column_stats = []

    for idx, col in enumerate(columns):
        col_name = col["name"]
        declared_type = normalize_declared_type(col["type"])
        values = [row[idx] for row in rows]
        null_count = sum(1 for val in values if val is None)
        non_null_values = [val for val in values if val is not None]
        non_null_count = len(non_null_values)

        type_mismatch_count = 0
        numeric_values = []
        for val in non_null_values:
            if declared_type in {"INTEGER", "REAL", "NUMERIC"}:
                if safe_float(val) is None:
                    type_mismatch_count += 1
                else:
                    numeric_values.append(float(val))
            elif declared_type == "TEXT":
                if not isinstance(val, str):
                    type_mismatch_count += 1
            elif declared_type == "BLOB":
                pass

        outliers = detect_outliers(numeric_values)
        outlier_summary = {
            "count": len(outliers),
            "samples": sorted(outliers)[:5],
        }

        stats = {
            "table": table_name,
            "column": col_name,
            "declared_type": declared_type,
            "notnull": col["notnull"],
            "primary_key": col["pk"],
            "row_count": len(rows),
            "null_count": null_count,
            "non_null_count": non_null_count,
            "type_mismatch_count": type_mismatch_count,
            "outlier_summary": outlier_summary,
        }

        if numeric_values:
            try:
                stats["mean"] = mean(numeric_values)
                stats["median"] = median(numeric_values)
                stats["stdev"] = stdev(numeric_values) if len(numeric_values) > 1 else 0.0
                stats["min"] = min(numeric_values)
                stats["max"] = max(numeric_values)
            except StatisticsError:
                stats["mean"] = stats["median"] = stats["stdev"] = 0.0
                stats["min"] = stats["max"] = None
        column_stats.append(stats)

    return column_stats


def write_report(report_path, summary):
    with open(report_path, "w", encoding="utf-8") as f:
        if not summary:
            f.write("No tables found in database.\n")
            return

        for table_name, stats in summary.items():
            f.write(f"TABLE: {table_name}\n")
            f.write("=" * (7 + len(table_name)) + "\n")
            for col in stats:
                f.write(f"Column: {col['column']}\n")
                f.write(f"  Declared type: {col['declared_type']}\n")
                f.write(f"  Primary key: {col['primary_key']}\n")
                f.write(f"  Not null: {col['notnull']}\n")
                f.write(f"  Rows: {col['row_count']}\n")
                f.write(f"  Null values: {col['null_count']}\n")
                f.write(f"  Non-null values: {col['non_null_count']}\n")
                f.write(f"  Type mismatch count: {col['type_mismatch_count']}\n")

                if col.get("mean") is not None:
                    f.write(f"  Mean: {col['mean']:.4f}\n")
                    f.write(f"  Median: {col['median']:.4f}\n")
                    f.write(f"  Stddev: {col['stdev']:.4f}\n")
                    f.write(f"  Min: {col['min']}\n")
                    f.write(f"  Max: {col['max']}\n")

                outlier = col["outlier_summary"]
                f.write(f"  Outlier count: {outlier['count']}\n")
                if outlier["samples"]:
                    f.write(f"  Example outliers: {outlier['samples']}\n")
                f.write("\n")
            f.write("\n")


def main():
    try:
        conn = sqlite3.connect(DB_PATH)
    except sqlite3.Error as err:
        print(f"Could not open database '{DB_PATH}': {err}")
        return

    cursor = conn.cursor()
    tables = get_tables(cursor)
    summary = {}

    for table_name in tables:
        summary[table_name] = summarize_table(cursor, table_name)

    write_report(REPORT_PATH, summary)
    conn.close()
    print(f"Data quality report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
