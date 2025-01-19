import kagglehub
import argparse
from pathlib import Path
import shutil

def download(output_path):
    # Set up argument parsing

    output_path.mkdir(parents=True, exist_ok=True)
    print("Output path:", output_path)

    # Download the dataset
    path = kagglehub.dataset_download("usmanafzaal/strawberry-disease-detection-dataset")

    print("Path to dataset files:", path)

    # Copy downloaded files to the output path
    for item in Path(path).iterdir():
        s = item
        d = output_path / item.name
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset to a specific path.")
    parser.add_argument(
        "--pathToDownload",
        type=str,
        required=True,
        help="Path to the directory where the dataset will be downloaded."
    )
    args = parser.parse_args()

    # Ensure the directory exists
    output_path = Path(args.pathToDownload)
    download(output_path)

    

