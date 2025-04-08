import pytesseract
from PIL import Image
import cv2
import easyocr
import numpy as np
import os
import re
import csv
from datetime import datetime
import sys
import pickle  # To save progress and avoid duplicate processing

# Ensure Tesseract is properly installed on your system and accessible
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\chandan.kumar\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'  # Update to your Tesseract path

# Define a lock file for ensuring that the process isn't edited or stopped
lock_file_path = "process_lock.pkl"

# Generate the current date for the CSV file name
current_date = datetime.now().strftime("%Y-%m-%d")
output_csv_path = f"{current_date}.csv"

# Function to create or check a lock file to prevent process tampering
def check_lock():
    if os.path.exists(lock_file_path):
        print("Process is already running or has been tampered with. Exiting.")
        sys.exit()
    else:
        with open(lock_file_path, 'wb') as lock_file:
            pickle.dump(True, lock_file)

# Function to release the lock
def release_lock():
    if os.path.exists(lock_file_path):
        os.remove(lock_file_path)

# Validate and check the folder path
def validate_folder_path(folder_path):
    if not os.path.isdir(folder_path):
        print("Error: Provided path is not a valid directory.")
        sys.exit()

    # Check for duplicate image paths
    processed_images_file = "processed_images.txt"
    if os.path.exists(processed_images_file):
        with open(processed_images_file, 'r') as f:
            processed_images = set(f.read().splitlines())
    else:
        processed_images = set()

    new_images = {f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.JPG'))}
    if new_images & processed_images:
        print("Warning: Some images have already been processed.")
        sys.exit()

    return new_images

# Save processed images to a file to avoid duplication
def save_processed_image(image_path):
    processed_images_file = "processed_images.txt"
    with open(processed_images_file, 'a') as f:
        f.write(image_path + '\n')

# Initialize EasyOCR to use the local models
model_storage_directory = r'C:\Users\Chandan.Kumar\Desktop\new_ocr\model_storage/'
reader = easyocr.Reader(['en'], model_storage_directory=model_storage_directory)

# Input folder path containing images
folder_path = input("Enter the folder path containing images: ")

# Lock the process
check_lock()

# Validate folder path and check for duplicate images
image_files = validate_folder_path(folder_path)

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, brightness_factor, contrast_factor):
    return cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor)

# Function to extract date with dynamic threshold adjustment
def extract_date_with_dynamic_threshold(image_path, save_images=False):
    test_img = cv2.imread(image_path)
    if test_img is None:
        print(f"Error: Image not found! {image_path}")
        return "Image not found"

    date_region = test_img[270:300, 200:350]
    result = reader.readtext(date_region, detail=1)

    if result:
        _, extracted_date, confidence = result[0]
        if confidence > 0.90 and len(extracted_date) == 3 and extracted_date.isalpha():
            return extracted_date

    date_gray = cv2.cvtColor(date_region, cv2.COLOR_BGR2GRAY)
    date_gray = adjust_brightness_contrast(date_gray, brightness_factor=15, contrast_factor=1.0)

    threshold_value = 150
    iteration = 0
    while True:
        _, date_thresh = cv2.threshold(date_gray, threshold_value, 255, cv2.THRESH_BINARY)
        if save_images:
            image_name = f"processed_image_iter_{iteration}.png"
            cv2.imwrite(image_name, date_thresh)

        result = reader.readtext(date_thresh, detail=1)
        if result:
            _, extracted_date, confidence = result[0]
            if confidence > 0.65 and confidence < 0.90 and len(extracted_date) == 3 and extracted_date.isalpha():
                return extracted_date
            elif confidence < 0.50:
                reduced_brightness = (date_gray * 0.75).astype("uint8")
                _, date_thresh_tesseract = cv2.threshold(reduced_brightness, 45, 255, cv2.THRESH_BINARY)
                tesseract_result = pytesseract.image_to_string(date_thresh_tesseract, config=r'--oem 3 --psm 6').strip()
                if len(tesseract_result) == 3 and tesseract_result.isalpha():
                    return tesseract_result

        threshold_value -= 5
        iteration += 1
        if threshold_value <= 25:
            return "Not Found"

# Function to clean and process MICR text
def clean_and_extract(text):
    if text and text[0].isalpha():
        text = text[1:]
    digits_only = re.sub(r'\D', '', text)
    cleaned_text = digits_only[:6] if len(digits_only) >= 6 else None

    if not cleaned_text:
        return None

    if len(cleaned_text) == 6:
        return cleaned_text
    elif len(cleaned_text) == 8:
        return cleaned_text[1:-1]
    elif len(cleaned_text) == 9:
        return cleaned_text[1:-2]
    elif len(cleaned_text) == 10:
        return cleaned_text[2:-2]
    elif len(cleaned_text) == 11:
        return cleaned_text[1:-4]
    else:
        return None

# Function to extract MICR text
def process_image_micr(image_path):
    image = Image.open(image_path)
    check_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    micr_img = check_img[320:380, 180:310]
    micr_text = pytesseract.image_to_string(micr_img, lang='mcr')
    return clean_and_extract(micr_text)

# Function to extract account number using EasyOCR
def extract_account_number(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return "Not Found"
    
    # Define the coordinates for extracting the account number region
    account_region = img[180:220, 80:290]
    account_gray = cv2.cvtColor(account_region, cv2.COLOR_BGR2GRAY)
    reduced_brightness = (account_gray * 0.85).astype("uint8")
    _, account_thresh = cv2.threshold(reduced_brightness, 55, 255, cv2.THRESH_BINARY)

    # Use EasyOCR to extract text
    results = reader.readtext(account_thresh)
    extracted_text = " ".join([result[1] for result in results])
    numeric_text = re.sub(r'[^0-9]', '', extracted_text)  # Retain only numeric characters
    
    if len(numeric_text) < 13:
        return "Not Found"
    return numeric_text

# Output dictionary for results
results = []

# Process all images in the folder
for file_name in image_files:
    file_path = os.path.join(folder_path, file_name)
    try:
        extracted_code = extract_date_with_dynamic_threshold(file_path, save_images=False)
        micr_text = process_image_micr(file_path)
        account_number = extract_account_number(file_path)

        result = reader.readtext(cv2.imread(file_path)[270:300, 200:350], detail=1)
        if result:
            _, easyocr_raw, confidence = result[0]
            if confidence > 0.80:
                extracted_code = easyocr_raw

        if len(extracted_code) > 3 or len(extracted_code) < 3:
            extracted_code = "Not Found"

        # Save the results for the current image immediately to the CSV file
        with open(output_csv_path, mode='a', newline='') as file:  # Open CSV in append mode
            writer = csv.DictWriter(file, fieldnames=["File Name", "Alpha Code", "Cheque No", "Account No"])
            if file.tell() == 0:  # Check if the file is empty to write the header
                writer.writeheader()
            writer.writerow({"File Name": file_name, "Alpha Code": extracted_code, "Cheque No": micr_text, "Account No": account_number})

        # Save processed image path to avoid future duplication
        save_processed_image(file_path)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        release_lock()
        sys.exit()

print(f"Results saved to {output_csv_path}")

# Release the lock
release_lock()
