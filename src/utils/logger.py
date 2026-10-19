import csv
import os
import json
from typing import Dict, Any

class CSVLogger:
    """Logs training metrics to a CSV file."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.initialized = False
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, metrics: Dict[str, Any]):
        file_exists = os.path.isfile(self.log_path)
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists or not self.initialized:
                writer.writeheader()
                self.initialized = True
            writer.writerow(metrics)

def save_config(config: dict, save_path: str):
    """Saves the configuration dictionary to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
