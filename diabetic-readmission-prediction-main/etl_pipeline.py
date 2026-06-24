"""
ETL Pipeline for Diabetic Readmission Prediction
Reads CSV files, cleans data, merges with mapping data, and loads into SQLite.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiabetesETLPipeline:
    """ETL pipeline for diabetic readmission prediction dataset."""
    
    def __init__(self, data_dir='.', db_name='diabetic_db.sqlite'):
        """
        Initialize the ETL pipeline.
        
        Args:
            data_dir: Directory containing CSV files
            db_name: Name of SQLite database to create
        """
        self.data_dir = Path(data_dir)
        self.db_name = db_name
        self.df = None
        self.mapping = None
        
    def extract(self):
        """Extract: Read CSV files from disk."""
        logger.info("Starting EXTRACT phase...")
        
        try:
            self.df = pd.read_csv(self.data_dir / "diabetic_data.csv")
            logger.info(f"✓ Loaded diabetic_data.csv: {self.df.shape}")
            
            self.mapping = pd.read_csv(self.data_dir / "IDS_mapping.csv")
            logger.info(f"✓ Loaded IDS_mapping.csv: {self.mapping.shape}")
            
        except FileNotFoundError as e:
            logger.error(f"Error reading files: {e}")
            raise
            
    def clean_data(self):
        """Clean: Handle missing values, duplicates, and data type conversions."""
        logger.info("Starting CLEAN phase...")
        
        # Replace '?' with NaN
        self.df.replace('?', np.nan, inplace=True)
        logger.info("✓ Replaced '?' with NaN")
        
        # Check missing values
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
        if len(missing_cols) > 0:
            logger.info("Missing values per column:")
            for col, pct in missing_cols.items():
                logger.info(f"  {col}: {pct:.2f}%")
        
        # Remove duplicate patients (keep first encounter)
        before = len(self.df)
        self.df = self.df.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first')
        logger.info(f"✓ Removed duplicate patients: {before} → {len(self.df)}")
        
        # Remove deaths and hospice discharges
        # Discharge disposition IDs: 11, 13, 14, 19, 20, 21 = deaths/hospice/other
        before = len(self.df)
        self.df = self.df[~self.df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
        logger.info(f"✓ Removed deaths/hospice discharges: {before} → {len(self.df)}")
        
        # Drop columns with excessive missing values
        self.df.drop(columns=['weight', 'payer_code'], inplace=True, errors='ignore')
        logger.info("✓ Dropped low-quality columns (weight, payer_code)")
        
    def create_target(self):
        """Create binary target variable for 30-day readmission."""
        logger.info("Creating target variable...")
        self.df['readmitted_30'] = (self.df['readmitted'] == '<30').astype(int)
        readmission_rate = self.df['readmitted_30'].mean() * 100
        logger.info(f"✓ Created readmitted_30: {readmission_rate:.2f}% positive rate")
        
    def map_admission_type(self):
        """Map admission_type_id to readable admission types."""
        logger.info("Mapping admission types...")
        
        if 'admission_type_id' not in self.df.columns:
            logger.info("✓ admission_type_id already processed, skipping")
            return
        
        # Define valid admission types
        valid_admissions = ['Emergency', 'Urgent', 'Elective', 'Newborn', 'Trauma Center']
        
        # Create mapping dictionary
        admission_map = self.mapping[
            self.mapping['description'].isin(valid_admissions)
        ]
        
        admission_dict = dict(zip(
            admission_map['admission_type_id'],
            admission_map['description']
        ))
        
        # Apply mapping and fill unknowns
        self.df['admission_type'] = self.df['admission_type_id'].map(admission_dict)
        self.df['admission_type'] = self.df['admission_type'].fillna('Unknown')
        
        # Drop the ID column
        self.df.drop('admission_type_id', axis=1, inplace=True)
        logger.info(f"✓ Mapped admission types: {self.df['admission_type'].nunique()} categories")
        
    def map_diagnosis_codes(self):
        """Map ICD-9 diagnosis codes to clinical categories."""
        logger.info("Mapping diagnosis codes...")
        
        def map_diag(code):
            """Convert ICD-9 code to clinical category."""
            if pd.isna(code):
                return 'Other'
            code = str(code)
            
            # Skip V and E codes
            if code.startswith('V') or code.startswith('E'):
                return 'Other'
            
            try:
                c = float(code)
                
                # ICD-9 ranges for common conditions
                if 390 <= c <= 459 or c == 785:
                    return 'Circulatory'
                if 460 <= c <= 519 or c == 786:
                    return 'Respiratory'
                if 520 <= c <= 579 or c == 787:
                    return 'Digestive'
                if c == 250:
                    return 'Diabetes'
                if 800 <= c <= 999:
                    return 'Injury'
                if 710 <= c <= 739:
                    return 'Musculoskeletal'
                if 580 <= c <= 629 or c == 788:
                    return 'Genitourinary'
                if 140 <= c <= 239:
                    return 'Neoplasms'
                if 380 <= c <= 389:
                    return 'Ear/Nose/Throat'
                if 320 <= c <= 359:
                    return 'Nervous'
                if 250 <= c <= 279:
                    return 'Endocrine'
                    
            except (ValueError, TypeError):
                pass
            
            return 'Other'
        
        self.df['diag_category'] = self.df['diag_1'].apply(map_diag)
        logger.info(f"✓ Mapped diagnosis codes: {self.df['diag_category'].nunique()} categories")
        logger.info(f"  Distribution: {self.df['diag_category'].value_counts().to_dict()}")
        
    def load_to_database(self):
        """Load: Save cleaned data to SQLite database."""
        logger.info("Starting LOAD phase...")
        
        try:
            engine = create_engine(f'sqlite:///{self.db_name}')
            
            # Save main table
            self.df.to_sql('encounters', engine, if_exists='replace', index=False)
            logger.info(f"✓ Loaded {len(self.df)} rows into 'encounters' table")
            
            # Create summary statistics table
            summary_stats = pd.DataFrame({
                'Metric': [
                    'Total Encounters',
                    'Unique Patients',
                    'Readmitted (30-day)',
                    'Readmission Rate (%)',
                    'Age Range',
                    'Avg Time in Hospital',
                    'Avg Num Medications'
                ],
                'Value': [
                    len(self.df),
                    self.df['patient_nbr'].nunique(),
                    self.df['readmitted_30'].sum(),
                    f"{self.df['readmitted_30'].mean() * 100:.2f}",
                    f"{self.df['age'].min()} to {self.df['age'].max()}",
                    f"{self.df['time_in_hospital'].mean():.1f}",
                    f"{self.df['num_medications'].mean():.1f}"
                ]
            })
            
            summary_stats.to_sql('summary_statistics', engine, if_exists='replace', index=False)
            logger.info(f"✓ Created summary_statistics table")
            
            # Create cohort analysis table
            cohort_analysis = self.df.groupby('admission_type').agg(
                total_encounters=('encounter_id', 'count'),
                unique_patients=('patient_nbr', 'nunique'),
                readmitted_count=('readmitted_30', 'sum'),
                readmission_rate=('readmitted_30', 'mean')
            ).reset_index()
            
            cohort_analysis.to_sql('cohort_analysis', engine, if_exists='replace', index=False)
            logger.info(f"✓ Created cohort_analysis table")
            
            logger.info(f"✓ Database saved to: {self.db_name}")
            
        except Exception as e:
            logger.error(f"Error loading to database: {e}")
            raise
            
    def run_pipeline(self):
        """Execute the complete ETL pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Diabetic Readmission ETL Pipeline")
        logger.info("=" * 60)
        
        try:
            self.extract()
            self.clean_data()
            self.create_target()
            self.map_admission_type()
            self.map_diagnosis_codes()
            self.load_to_database()
            
            logger.info("=" * 60)
            logger.info("✓ ETL Pipeline completed successfully!")
            logger.info("=" * 60)
            
            return self.df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
            
    def get_summary(self):
        """Print summary statistics about the processed data."""
        if self.df is None:
            logger.warning("No data loaded. Run pipeline first.")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("DATA SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Shape: {self.df.shape}")
        logger.info(f"Memory: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"\nColumns ({len(self.df.columns)}):")
        for col in self.df.columns:
            logger.info(f"  {col}: {self.df[col].dtype}")
        logger.info("=" * 60 + "\n")


def main():
    """Main entry point for the ETL pipeline."""
    
    # Initialize and run pipeline
    pipeline = DiabetesETLPipeline(
        data_dir='.',  # Current directory
        db_name='diabetic_db.sqlite'
    )
    
    # Run the ETL pipeline
    df = pipeline.run_pipeline()
    
    # Print summary
    pipeline.get_summary()
    
    return df, pipeline


if __name__ == "__main__":
    df, pipeline = main()
