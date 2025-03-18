import os
from datetime import datetime

def get_latest_processed_file(data_path, filename_starts_with="processed_data_"):
    """Find the latest file based on the date in the filename starting with filename_starts_with in the given directory.
    Returns the full file path and the extracted date.
    """

    files = [f for f in os.listdir(data_path) if f.startswith(filename_starts_with)]

    if not files:
        print(f"No files found in {data_path} starting with {filename_starts_with}.")
        return None, None

    try:
        latest_file = max(
            files,
            key=lambda x: datetime.strptime(
                os.path.splitext(x)[0].split("_")[-1], "%Y-%m-%d"
            ),
        )
        latest_date = datetime.strptime(
            os.path.splitext(latest_file)[0].split("_")[-1], "%Y-%m-%d"
        ).date()
    except ValueError:
        print("Error: Some filenames do not match the expected format.")
        return None, None

    return os.path.join(data_path, latest_file), latest_date


# Example usage
# latest_file_path, latest_date = get_latest_processed_file(r'E:\My_Github\fr-fuel-price-tracking\data')
# print("Latest processed file path:", latest_file_path)
# print("Latest date found:", latest_date)
