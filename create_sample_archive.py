import os
from pathlib import Path
from tests.conftest import make_test_archive

def main():
    dest_dir = Path("data/archives")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating sample archive...")
    # Generate 500 ticks of test data directly to the destination directory
    archive_path = make_test_archive(dest_dir, ticks_per_symbol=500, symbols=["NIFTY26JUNFUT"])
    print(f"Sample archive successfully created at: {archive_path.resolve()}")

if __name__ == "__main__":
    main()
