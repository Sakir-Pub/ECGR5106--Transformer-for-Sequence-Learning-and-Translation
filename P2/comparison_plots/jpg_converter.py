import os
import sys
from PIL import Image

def convert_png_to_jpeg(source_folder, destination_folder='jpeg_converted'):
    """
    Converts all PNG images in source_folder to jpeg format and saves them in destination_folder.
    
    Args:
        source_folder (str): Path to folder containing PNG images
        destination_folder (str): Path to folder where jpeg images will be saved
    """
    # Check if source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist")
        return False
    
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")
    
    # Get all PNG files in source folder
    png_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.png')]
    
    if not png_files:
        print(f"No PNG files found in '{source_folder}'")
        return False
    
    # Convert each PNG file to jpeg
    conversion_count = 0
    for png_file in png_files:
        try:
            # Open PNG image
            png_path = os.path.join(source_folder, png_file)
            img = Image.open(png_path)
            
            # Create jpeg filename
            jpeg_filename = os.path.splitext(png_file)[0] + '.jpeg'
            jpeg_path = os.path.join(destination_folder, jpeg_filename)
            
            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                img = background
            
            # Save as jpeg
            img.save(jpeg_path, 'JPEG', quality=95)
            conversion_count += 1
            print(f"Converted: {png_file} -> {jpeg_filename}")
            
        except Exception as e:
            print(f"Error converting {png_file}: {e}")
    
    print(f"\nConversion complete. {conversion_count} of {len(png_files)} PNG files converted to jpeg.")
    return True

if __name__ == "__main__":
    # Use command line argument for source folder if provided, otherwise use current directory
    source = "/home/anabil/Development/PhD Courses/ECGR 5106/HW5/P2/comparison_plots"
    convert_png_to_jpeg(source)