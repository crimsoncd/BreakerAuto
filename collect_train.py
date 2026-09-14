import argparse
import json
import os
from PIL import Image


def process_dataset(source_dir, output_dir, start_index):
    os.makedirs(output_dir, exist_ok=True)
    current_idx = start_index

    # Recursively traverse source_dir to find metadata.json files
    for root, _, files in os.walk(source_dir):
        if "metadata.json" in files:
            json_path = os.path.join(root, "metadata.json")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            width, height = data["image_size"]

            for elem in data.get("elements", []):
                name = elem.get("name", "")
                # Skip background elements
                if name == "background":
                    continue

                # Format object name (replace underscores with spaces)
                clean_name = name.replace("_", " ")
                bbox = elem["bbox"]  # [xmin, ymin, xmax, ymax] in 0-1000 scale
                layer = elem["layer"]
                rel_path = elem["path"]

                # Resolve element image path
                img_path = os.path.join(source_dir, rel_path)
                if not os.path.exists(img_path):
                    img_path = os.path.join(root, os.path.basename(rel_path))

                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found: {img_path}")
                    continue

                # Convert normalized coordinates (0-1000) to pixel coordinates
                # xmin = int(bbox[0] / 1000.0 * width)
                # ymin = int(bbox[1] / 1000.0 * height)
                # xmax = int(bbox[2] / 1000.0 * width)
                # ymax = int(bbox[3] / 1000.0 * height)

                # target_w = max(1, xmax - xmin)
                # target_h = max(1, ymax - ymin)

                # # Create full image-sized transparent black RGBA canvas
                # canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

                # # Load and resize element image to target bounding box size
                # elem_img = Image.open(img_path).convert("RGBA")
                # elem_img = elem_img.resize(
                #     (target_w, target_h), Image.Resampling.LANCZOS
                # )

                # # Paste element image into canvas at the specified bbox coordinates
                # canvas.paste(elem_img, (xmin, ymin), elem_img)

                # Generate formatted 4-digit filename
                file_id = f"{current_idx:04d}"
                out_img_path = os.path.join(output_dir, f"{file_id}.png")
                out_txt_path = os.path.join(output_dir, f"{file_id}.txt")

                image = Image.open(img_path).convert("RGBA")
                image.save(out_img_path)

                # Save output image
                # canvas.save(out_img_path)

                # Save caption text file
                # caption = f"<object>{clean_name}</object><bbox>{bbox}</bbox><layer>{layer}</layer>"
                # with open(out_txt_path, "w", encoding="utf-8") as txt_file:
                #     txt_file.write(caption)

                current_idx += 1

    print("Completed with index:" , current_idx - 1)
    print("Next index for future runs:", current_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process elements into RGBA full-sized canvases and format caption TXT files."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to source directory containing subfolders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to destination output directory.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="Starting integer index for output filename counter.",
    )

    args = parser.parse_args()
    process_dataset(args.source_dir, args.output_dir, args.start_index)