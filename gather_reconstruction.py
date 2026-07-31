import argparse
import shutil
from pathlib import Path


def process_images(source_dir, output_dir):
    source = Path(source_dir)
    output = Path(output_dir)

    # Create the output directory if it doesn't exist
    output.mkdir(parents=True, exist_ok=True)

    # Iterate through all items in the source directory
    for folder in source.iterdir():
        # Make sure we are only processing directories
        if not folder.is_dir():
            continue

        # Extract the label (characters before the first '_')
        if "_" not in folder.name:
            print(f"Skipping folder (no '_' found): {folder.name}")
            continue

        label = folder.name.split("_")[0]

        # Find all files containing "reconstruction" in their name
        # We search for any extension, but you can restrict it if needed (e.g., *reconstruction*.png)
        matching_images = sorted(list(folder.glob("*reconstruction*")))

        if not matching_images:
            print(f"No reconstruction image found in: {folder.name}")
            continue

        # Get the last image after sorting
        target_image = matching_images[-1]

        # Define the new filename and destination path
        # Using the target image's original suffix (e.g., .png or .jpg)
        new_filename = f"{label}-reconstruction{target_image.suffix}"
        destination = output / new_filename

        # Copy the file
        shutil.copy2(target_image, destination)
        print(f"Copied: {target_image.name} -> {destination.name}")


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Process and copy reconstruction images."
    )
    parser.add_argument(
        "--source", required=True, help="Path to the source directory"
    )
    parser.add_argument(
        "--output", required=True, help="Path to the output directory"
    )

    args = parser.parse_args()

    process_images(args.source, args.output)