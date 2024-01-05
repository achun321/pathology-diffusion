from PIL import Image
import os
import json
from tqdm import tqdm

def resize_images(input_dir, output_dir, target_size=(1440, 1440)):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all files in the input directory
    metadata_file = "./test/metadata.jsonl"
    with open(metadata_file, "r") as f:
        metadata_list = [json.loads(line) for line in f]

    # For each entry in metadata -- first 200 images in test dataset
    for i in tqdm(range(200)):
        metadata = metadata_list[i]
        # Use the "text" field as prompt
        file_name = metadata["file_name"]

        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        try: 
            # Open the image using Pillow
            img = Image.open(input_path)

            # Resize the image to the target size
            img_resized = img.resize(target_size, Image.LANCZOS)

            # Save the resized image to the output directory
            img_resized.save(output_path)
        except: 
            pass



if __name__ == "__main__":
    # Specify your input and output directories
    input_directory = "./test"
    output_directory = "./test-processed"

    # Specify the target size
    target_size = (512, 512)

    # Resize images and save them to the output directory
    resize_images(input_directory, output_directory, target_size)