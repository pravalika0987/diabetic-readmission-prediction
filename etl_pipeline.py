import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DiabetesETL:
    def __init__(self, db_name='diabetic_db.sqlite'):
        self.db_name = db_name
        self.diabetic_data = None
        self.ids_mapping = None
        self.merged_data = None

    def extract(self):
        """Read CSV files"""
        logger.info("Starting extraction phase...")
        try:
            self.diabetic_data = pd.read_csv('diabetic_data.csv')
            logger.info(f"Loaded diabetic_data.csv: {self.diabetic_data.shape[0]} rows, {self.diabetic_data.shape[1]} columns")

            self.ids_mapping = pd.read_csv('IDS_mapping.csv')
            logger.info(f"Loaded IDS_mapping.csv: {self.ids_mapping.shape[0]} rows, {self.ids_mapping.shape[1]} columns")
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise

    def transform(self):
        """Clean and prepare data"""
        logger.info("Starting transformation phase...")

        # Clean diabetic data
        logger.info("Cleaning diabetic_data...")
        self.diabetic_data.drop_duplicates(inplace=True)
        self.diabetic_data.dropna(thresh=len(self.diabetic_data.columns) * 0.5, inplace=True)

        for col in self.diabetic_data.columns:
            if self.diabetic_data[col].dtype == 'object':
                self.diabetic_data[col] = self.diabetic_data[col].str.strip()

        logger.info(f"After cleaning: {self.diabetic_data.shape[0]} rows remaining")

        # Clean ID mapping
        logger.info("Cleaning IDS_mapping...")
        self.ids_mapping.drop_duplicates(inplace=True)
        self.ids_mapping.dropna(inplace=True)

        for col in self.ids_mapping.columns:
            if self.ids_mapping[col].dtype == 'object':
                self.ids_mapping[col] = self.ids_mapping[col].str.strip()

        logger.info(f"ID mapping cleaned: {self.ids_mapping.shape[0]} rows")

    def merge(self):
        """Merge datasets"""
        logger.info('Starting merge phase...')
        self.ids_mapping['admission_type_id'] = pd.to_numeric(self.ids_mapping['admission_type_id'], errors='coerce')

        # Identify common columns for merge
        common_cols = list(set(self.diabetic_data.columns) & set(self.ids_mapping.columns))

        if not common_cols:
            logger.warning("No common columns found. Attempting merge on index or first ID-like column...")
            if 'patient_id' in self.diabetic_data.columns and 'patient_id' in self.ids_mapping.columns:
                merge_key = 'patient_id'
            elif 'id' in self.diabetic_data.columns and 'id' in self.ids_mapping.columns:
                merge_key = 'id'
            else:
                merge_key = common_cols[0] if common_cols else None
        else:
            merge_key = common_cols[0]

        if merge_key:
            logger.info(f"Merging on column: {merge_key}")
            self.merged_data = pd.merge(
                self.diabetic_data,
                self.ids_mapping,
                on=merge_key,
                how='left'
            )
        else:
            logger.warning("Could not identify merge key. Concatenating dataframes...")
            self.merged_data = pd.concat([self.diabetic_data, self.ids_mapping], axis=1)

        logger.info(f"Merged data shape: {self.merged_data.shape[0]} rows, {self.merged_data.shape[1]} columns")

    def load(self):
        """Load data into SQLite database"""
        logger.info("Starting load phase...")

        try:
            conn = sqlite3.connect(self.db_name)
            self.merged_data.to_sql('diabetic_records', conn, if_exists='replace', index=False)
            conn.close()
            logger.info(f"Successfully loaded data into {self.db_name}")
            logger.info(f"Table 'diabetic_records' created with {len(self.merged_data)} records")
        except Exception as e:
            logger.error(f"Error during load: {e}")
            raise

    def run(self):
        """Execute the full ETL pipeline"""
        logger.info("=" * 50)
        logger.info("Starting ETL Pipeline")
        logger.info("=" * 50)

        try:
            self.extract()
            self.transform()
            self.merge()
            self.load()

            logger.info("=" * 50)
            logger.info("ETL Pipeline completed successfully!")
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            raise


if __name__ == '__main__':
    etl = DiabetesETL()
    etl.run()
