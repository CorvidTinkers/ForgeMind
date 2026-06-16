import time
import requests
import json
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000/api/v1"

def print_header(text):
    print(f"\n{'='*50}\n{text}\n{'='*50}")

def wait_for_status(url, target_status, max_retries=60):
    for i in range(max_retries):
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status")
            print(f"Current status: {status}")
            if status == target_status:
                return data
            if status == "failed":
                print(f"Task failed: {json.dumps(data, indent=2)}")
                sys.exit(1)
        time.sleep(2)
    print("Timed out waiting for status change.")
    sys.exit(1)

def main():
    print_header("ForgeMind E2E Test (Phase 1 & 2)")

    # 1. Create a dummy CSV dataset
    print("Creating test dataset (test_data.csv)...")
    csv_path = Path("test_data.csv")
    csv_path.write_text("question,answer\nWhat is gravity?,Gravity is a force.\nWho is AI?,AI is artificial intelligence.\n")

    # 2. Upload to trigger Phase 1
    print("\nUploading dataset to trigger Phase 1 (Analysis)...")
    with open(csv_path, "rb") as f:
        resp = requests.post(f"{BASE_URL}/datasets/upload", files={"file": f})
    
    if resp.status_code != 202:
        print(f"Upload failed: {resp.text}")
        sys.exit(1)
        
    dataset_id = resp.json()["id"]
    print(f"Dataset ID: {dataset_id}")

    # 3. Wait for Analysis to complete
    print("\nWaiting for Phase 1 to finish (Status: 'analyzed')...")
    wait_for_status(f"{BASE_URL}/datasets/{dataset_id}", "analyzed")

    # 4. Trigger Phase 2 (Augmentation)
    print("\nTriggering Phase 2 (Data Augmentation)...")
    resp = requests.post(f"{BASE_URL}/datasets/{dataset_id}/augment")
    if resp.status_code != 202:
        print(f"Augment trigger failed: {resp.text}")
        sys.exit(1)
        
    run_id = resp.json()["run_id"]
    print(f"Augmentation Run ID: {run_id}")

    # 5. Wait for Augmentation to complete
    print("\nWaiting for Phase 2 to finish (this requires Ollama to generate data)...")
    wait_for_status(f"{BASE_URL}/augmentation/{run_id}", "completed", max_retries=120)

    # 6. Fetch the report
    print("\nFetching Augmentation Report...")
    resp = requests.get(f"{BASE_URL}/augmentation/{run_id}/report")
    report = resp.json()
    print(json.dumps(report, indent=2))

    # 7. Download Final Alpaca Dataset
    print("\nDownloading Final Training Dataset (Alpaca JSON)...")
    resp = requests.get(f"{BASE_URL}/training-dataset/{dataset_id}/download?format=alpaca&file_format=json")
    if resp.status_code == 200:
        final_data = resp.json()
        output_path = Path("augmented_alpaca.json")
        output_path.write_text(json.dumps(final_data, indent=2))
        print(f"SUCCESS! Saved {len(final_data)} augmented samples to {output_path.absolute()}")
    else:
        print(f"Download failed: {resp.text}")

if __name__ == "__main__":
    main()
