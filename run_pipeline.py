import logging
import os
import subprocess
import sys

SCRIPT_NAMES = ["etl_pipeline.py", "data_quality.py", "feature_engineering.py", "train_model.py"]
LOG_FILE = "run_pipeline.log"


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_script(script_path):
    logging.info(f"Starting {os.path.basename(script_path)}")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        logging.info(result.stdout.strip())
    if result.stderr:
        logging.error(result.stderr.strip())

    if result.returncode != 0:
        logging.error(f"{os.path.basename(script_path)} failed with exit code {result.returncode}")
        raise RuntimeError(f"Script failed: {script_path}")

    logging.info(f"Finished {os.path.basename(script_path)} successfully")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, LOG_FILE)
    setup_logging(log_path)

    logging.info("Pipeline started")
    for script_name in SCRIPT_NAMES:
        script_path = os.path.join(base_dir, script_name)
        if not os.path.exists(script_path):
            logging.error(f"Script not found: {script_path}")
            sys.exit(1)

        try:
            run_script(script_path)
        except Exception as exc:
            logging.exception("Pipeline stopped due to error")
            sys.exit(1)

    logging.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
