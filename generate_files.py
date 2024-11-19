import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from pyarabic.araby import strip_tatweel
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar, normalize_alef_ar, normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar

# Paths
input_csv_path = "OCR_dataset.csv"  # Replace with your filtered dataset file
output_folder = "/content/ocr_training_data"  # Folder to save images and ground truth files
font_paths = [
    "/usr/share/fonts/truetype/arial.ttf",  # Default font
    "/usr/share/fonts/truetype/AL-Mohanad.ttf",  # Al-Mohanad font
    "/usr/share/fonts/truetype/calibri.ttf"  # Calibri font
]
font_size = 24

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Step 1: Normalize text
def normalize_text(text):
    text = strip_tatweel(text)                 # Removing tatweel (elongations)
    text = normalize_unicode(text)            # Normalizes Unicode characters
    text = normalize_alef_ar(text)            # Normalizes Alef characters
    text = normalize_alef_maksura_ar(text)    # Converts Alef Maksura to Yeh
    text = normalize_teh_marbuta_ar(text)     # Normalizes Teh Marbuta characters
    return text

# Step 2: Remove diacritics
def remove_diacritics(text):
    return dediac_ar(text)

# Function to add scan effects (noise, blur, skew)
def add_scan_effects(image):
    # Add random noise
    noise = Image.effect_noise(image.size, random.randint(1, 10))
    image = Image.blend(image, noise.convert("RGB"), alpha=0.2)

    # Apply Gaussian blur
    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    # Skew the image
    angle = random.uniform(-2, 2)  # Small skew angle
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False)

    return image

# Load the dataset
df = pd.read_csv(input_csv_path, encoding='utf-8')

# Process each article
file_counter = 0
for index, row in df.iterrows():
    article = row['text']

    # Normalize and clean the text
    cleaned_text = normalize_text(article)  # Normalize text
    cleaned_text = remove_diacritics(cleaned_text)  # Remove diacritics

    # Split cleaned text into sentences of 8 words
    words = cleaned_text.split()
    for i in range(0, len(words), 8):
        sentence = " ".join(words[i:i + 8])  # Take 8 words at a time

        # Reshape and reorder the sentence for Arabic rendering
        #reshaped_sentence = reshape(sentence)
        #bidi_sentence = get_display(reshaped_sentence)

        # Generate image for this sentence
        image = Image.new('RGB', (800, 100), color=(255, 255, 255))  # Create a blank white image
        draw = ImageDraw.Draw(image)

        try:
            # Randomly select a font
            font = ImageFont.truetype(random.choice(font_paths), font_size)
        except Exception as e:
            print(f"Font not found or invalid: {e}")
            exit()

        # Add text to the image
        draw.text((10, 10), sentence, fill=(0, 0, 0), font=font)

        # Apply scan effects (noise, blur, skew)
        #image = add_scan_effects(image)

        # Save the image in .tif format
        image_filename = f"{file_counter}.tif"
        image_path = os.path.join(output_folder, image_filename)
        image.save(image_path)

        # Save the corresponding .gt.txt file (original cleaned text)
        gt_filename = f"{file_counter}.gt.txt"
        gt_path = os.path.join(output_folder, gt_filename)
        with open(gt_path, "w", encoding="utf-8") as gt_file:
            gt_file.write(sentence)

        file_counter += 1
        if file_counter == 100:  # Limit to 100 samples
            break
    if file_counter == 100:
        break

print(f"Generated {file_counter} image-text pairs in the folder: {output_folder}")
