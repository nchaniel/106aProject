# because Nathan C uploads heic photos
from pillow_heif import register_heif_opener
from PIL import Image
import os

register_heif_opener()

input_folder = "heic_images"
output_folder = "jpg_images"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".heic"):
        path = os.path.join(input_folder, filename)

        image = Image.open(path)

        new_name = os.path.splitext(filename)[0] + ".jpg"
        save_path = os.path.join(output_folder, new_name)

        image.save(save_path, "JPEG")

        print(f"Converted: {filename}")

