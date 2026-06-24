import sqlite3
import os

DB_PATH = "diabetic_db.sqlite"
SOURCE_TABLE = "diabetic_records"
TARGET_TABLE = "ml_ready_features"
BATCH_SIZE = 5000

TEXT_EXCLUDE = {"readmitted"}


def quote_identifier(name):
    return f'"{name.replace("\"", "\"\"")}"'


def get_table_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({quote_identifier(table_name)})")
    return [
        {
            "name": row[1],
            "type": row[2].upper().strip() if row[2] else "TEXT",
        }
        for row in cursor.fetchall()
    ]


def build_category_map(cursor, column_name):
    cursor.execute(
        f"SELECT DISTINCT {quote_identifier(column_name)} FROM {quote_identifier(SOURCE_TABLE)} ORDER BY {quote_identifier(column_name)}"
    )
    values = [row[0] for row in cursor.fetchall()]
    mapping = {None: 0}
    next_id = 1
    for value in values:
        if value is None:
            continue
        mapping[value] = next_id
        next_id += 1
    return mapping


def create_target_table(cursor, columns, categorical_columns):
    parts = []
    for col in columns:
        if col["name"] == "readmitted":
            continue
        if col["type"] == "TEXT" and col["name"] in categorical_columns:
            parts.append(f"{quote_identifier(col['name'] + '_id')} INTEGER")
        else:
            parts.append(f"{quote_identifier(col['name'])} {col['type']}")

    parts.append("total_visits INTEGER")
    parts.append("avg_labs_per_day REAL")
    parts.append("meds_per_diagnosis REAL")
    parts.append("readmitted_binary INTEGER")
    parts.append("readmitted_label INTEGER")

    columns_sql = ",\n    ".join(parts)
    cursor.execute(f"DROP TABLE IF EXISTS {quote_identifier(TARGET_TABLE)}")
    cursor.execute(
        f"CREATE TABLE {quote_identifier(TARGET_TABLE)} (\n    {columns_sql}\n)"
    )


def normalize_row(row, columns, cat_maps):
    new_row = []
    row_dict = {col["name"]: row[idx] for idx, col in enumerate(columns)}

    for col in columns:
        name = col["name"]
        value = row_dict[name]
        if name == "readmitted":
            continue
        if col["type"] == "TEXT" and name in cat_maps:
            new_row.append(cat_maps[name].get(value, 0))
        else:
            new_row.append(value)

    num_outpatient = row_dict.get("number_outpatient") or 0
    num_emergency = row_dict.get("number_emergency") or 0
    num_inpatient = row_dict.get("number_inpatient") or 0
    time_in_hospital = row_dict.get("time_in_hospital") or 0
    num_lab_procedures = row_dict.get("num_lab_procedures") or 0
    num_medications = row_dict.get("num_medications") or 0
    number_diagnoses = row_dict.get("number_diagnoses") or 0

    total_visits = num_outpatient + num_emergency + num_inpatient
    avg_labs_per_day = float(num_lab_procedures) / time_in_hospital if time_in_hospital else 0.0
    meds_per_diagnosis = float(num_medications) / number_diagnoses if number_diagnoses else 0.0

    readmitted_value = row_dict.get("readmitted")
    readmitted_binary = 0 if readmitted_value == "NO" else 1
    if readmitted_value == "NO":
        readmitted_label = 0
    elif readmitted_value == ">30":
        readmitted_label = 1
    elif readmitted_value == "<30":
        readmitted_label = 2
    else:
        readmitted_label = 0

    new_row.extend([total_visits, avg_labs_per_day, meds_per_diagnosis, readmitted_binary, readmitted_label])
    return new_row


def main():
    if not os.path.exists(DB_PATH):
        print(f"Database file not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    columns = get_table_columns(cursor, SOURCE_TABLE)
    categorical_columns = [col["name"] for col in columns if col["type"] == "TEXT" and col["name"] not in TEXT_EXCLUDE]
    cat_maps = {name: build_category_map(cursor, name) for name in categorical_columns}

    create_target_table(cursor, columns, categorical_columns)

    insert_columns = [
        quote_identifier(col["name"] + ("_id" if col["type"] == "TEXT" and col["name"] in categorical_columns else ""))
        for col in columns
        if col["name"] != "readmitted"
    ]
    insert_columns.extend(["total_visits", "avg_labs_per_day", "meds_per_diagnosis", "readmitted_binary", "readmitted_label"])
    placeholders = ", ".join(["?"] * len(insert_columns))
    insert_sql = f"INSERT INTO {quote_identifier(TARGET_TABLE)} ({', '.join(insert_columns)}) VALUES ({placeholders})"

    cursor.execute(f"SELECT * FROM {quote_identifier(SOURCE_TABLE)}")
    batch = []
    row_count = 0
    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break
        for row in rows:
            batch.append(normalize_row(row, columns, cat_maps))
            row_count += 1
        cursor.executemany(insert_sql, batch)
        conn.commit()
        batch = []

    print(f"Created {TARGET_TABLE} with {row_count} rows")
    conn.close()


if __name__ == "__main__":
    main()
