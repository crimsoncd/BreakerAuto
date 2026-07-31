import argparse
import os
import sys
import json
import glob
import shutil

from PIL import Image

def find_file_robustly(src_dir, planned_path, fallback_pattern):
    """
    Attempts to find a file using its planned path from the JSON. 
    If that path doesn't exist (e.g., due to moving directories), 
    it searches the subfolder by filename or fallback pattern.
    """
    # 1. Try absolute/planned path directly
    if planned_path and os.path.exists(planned_path):
        return planned_path
    
    # 2. Try looking for the exact filename within the current source subfolder
    if planned_path:
        base_name = os.path.basename(planned_path)
        for root, _, files in os.walk(src_dir):
            if base_name in files:
                return os.path.join(root, base_name)
                
    # 3. Fallback to searching by a pattern (e.g., *background*.png)
    matches = glob.glob(os.path.join(src_dir, "**", fallback_pattern), recursive=True)
    if matches:
        return matches[0]
        
    return None

def process_dataset(input_dir, output_root, collection_code):
    print(f"Starting dataset cleanup for collection: {collection_code}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Root: {output_root}\n")

    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # List and sort all subfolders
    subfolders = sorted(os.listdir(input_dir))
    
    for folder_name in subfolders:
        folder_path = os.path.join(input_dir, folder_name)
        
        # Check if it's a directory and starts with a 3-digit index
        if not os.path.isdir(folder_path) or not (len(folder_name) >= 3 and folder_name[:3].isdigit()):
            continue
            
        index = folder_name[:3]
        new_folder_name = f"{collection_code}{index}"
        
        # Define the target paths
        # Format: output_root/A/A000/
        target_dir = os.path.join(output_root, collection_code, new_folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # Base relative path format: A/A000
        relative_folder_path = f"{collection_code}/{new_folder_name}"
        
        print(f"Processing folder [{folder_name}] -> Target: [{relative_folder_path}]")
        
        # --- 1. Find and copy the reconstruction image ---
        recon_matches = glob.glob(os.path.join(folder_path, "**", "*reconstruction*.png"), recursive=True)
        if not recon_matches:
            print(f"  Warning: No 'reconstruction' PNG found in {folder_name}. Skipping folder.")
            continue
            
        src_recon_path = recon_matches[0]
        recon_filename = f"{new_folder_name}.png"
        dest_recon_path = os.path.join(target_dir, recon_filename)
        shutil.copy(src_recon_path, dest_recon_path)

        recon_img = Image.open(src_recon_path)
        recon_width, recon_height = recon_img.size
        
        recon_relative_path = f"{relative_folder_path}/{recon_filename}"
        
        # --- 2. Find and parse the final_shipped JSON ---
        json_matches = glob.glob(os.path.join(folder_path, "**", "*final_shipped*.json"), recursive=True)
        if not json_matches:
            print(f"  Warning: No 'final_shipped' JSON found in {folder_name}. Skipping metadata extraction.")
            continue
            
        src_json_path = json_matches[0]
        with open(src_json_path, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
            
        image_size = old_data.get("image_size", [1024, 1024])

        if image_size != [recon_width, recon_height]:
            # print(f"  Warning: Image size mismatch for {folder_name}. JSON size: {image_size}, Actual size: {[recon_width, recon_height]}. Using actual size.")
            image_size = [recon_width, recon_height]

        new_elements = []
        
        # --- 3. Process Background Element ---
        bg_info = old_data.get("background", {})
        bg_src_path = bg_info.get("image_path")
        
        # Find background file location robustly
        actual_bg_path = find_file_robustly(folder_path, bg_src_path, "*background*.png")
        
        if actual_bg_path:
            bg_filename = f"{new_folder_name}_background.png"
            dest_bg_path = os.path.join(target_dir, bg_filename)
            shutil.copy(actual_bg_path, dest_bg_path)
            
            new_elements.append({
                "name": "background",
                "bbox": [0, 0, 1000, 1000],
                "layer": 0,
                "path": f"{relative_folder_path}/{bg_filename}"
            })
        else:
            print(f"  Warning: Background image not found for {folder_name}")

        # --- 4. Process Foreground Elements ---
        elements = old_data.get("elements", [])
        for elem in elements:
            elem_name = elem.get("name")
            bbox = elem.get("bbox")
            layer = elem.get("depth_rank", 1)
            layer_path = elem.get("layer_path")
            
            # Find layer file location robustly
            actual_layer_path = find_file_robustly(folder_path, layer_path, f"*_{elem_name}.png")
            
            if actual_layer_path:
                elem_filename = f"{new_folder_name}_{elem_name}.png"
                dest_elem_path = os.path.join(target_dir, elem_filename)
                shutil.copy(actual_layer_path, dest_elem_path)
                
                new_elements.append({
                    "name": elem_name,
                    "bbox": bbox,
                    "layer": layer,
                    "path": f"{relative_folder_path}/{elem_filename}"
                })
            else:
                print(f"  Warning: Layer image for '{elem_name}' not found in {folder_name}")
                
        # --- 5. Generate and save new metadata.json ---
        metadata = {
            "image_path": recon_relative_path,
            "image_size": image_size,
            "elements": new_elements
        }
        
        metadata_path = os.path.join(target_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
    print("\nDataset cleaning and sorting completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort layer-decomposition run folders into a cleaned dataset."
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory containing run folders, e.g. runsreal/<collection>/"
    )
    parser.add_argument(
        "--output", required=True,
        help="Root directory to write the cleaned dataset into"
    )
    parser.add_argument(
        "--code", default="C",
        help="Collection code prefix (default: C). Folders are renamed to e.g. C000."
    )
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.code)