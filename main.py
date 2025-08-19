from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.data as tfd
import io
import csv
import os
import tempfile
import subprocess
import json
from typing import List, Dict, Any
import logging
import pandas as pd

# Load back
with open("vocab.json", "r") as f:
    vocab_loaded = json.load(f)

# Recreate StringLookup
char_to_num = layers.StringLookup(vocabulary=vocab_loaded, mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(),
                                  mask_token=None, invert=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spanish OCR API", description="API for Spanish text recognition from images")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model and configurations
inference_model = None
# char_to_num = None
# num_to_char = None
n_classes=50
IMG_WIDTH = 200
IMG_HEIGHT = 50
MAX_LABEL_LENGTH = None
# AUTOTUNE
AUTOTUNE = tfd.AUTOTUNE
# Batch Size
BATCH_SIZE = 16

# Character mapping for corrections
# CHARACTER_MAPPING = {
#     'в': 'o',
#     'д': 'ñ',
#     'б': 'i',
#     'В': 'e',
#     'а': 'a'
# }

class CTCLayer(layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost
    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_len = tf.cast(tf.shape(y_pred)[1], dtype='int64') * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_len = tf.cast(tf.shape(y_true)[1], dtype='int64') * tf.ones(shape=(batch_len, 1), dtype='int64')
        loss = self.loss_fn(y_true, y_pred, input_len, label_len)
        self.add_loss(loss)
        return y_pred


def load_model_and_setup():
    """Load the trained OCR model and setup character mappings"""
    global inference_model, char_to_num, num_to_char, MAX_LABEL_LENGTH
    
    try:
        # Define unique characters (update this with your actual character set)
        # unique_chars = {'e', 'j', 'Q', 'z', 'v', 'A', 'L', 't', 'V', 'O', 'c', 'q', 'l', 'a', 'ñ', 'B', 'P', ',', 'H', 'C', 'M', 'G', 's', 'r', 'T', 'd', 'g', 'p', 'D', 'S', 'N', 'b', 'm', 'u', 'o', 'f', 'I', 'x', 'R', 'y', 'n', 'i', '-', 'F', 'E', 'h'}
        
        # # Character to numeric mapping
        # char_to_num = layers.StringLookup(
        #     vocabulary=list(unique_chars),
        #     mask_token=None
        # )
        
        # # Reverse mapping
        # num_to_char = layers.StringLookup(
        #     vocabulary=char_to_num.get_vocabulary(),
        #     mask_token=None,
        #     invert=True
        # )
        
        # Load your trained model
        model_path = '../Submission/ocr_model_NEW.h5'  # Update with your model path
        if os.path.exists(model_path):
            full_model = keras.models.load_model(model_path, compile=False, custom_objects={'CTCLayer': CTCLayer})
            # Create inference model
            inference_model = keras.Model(
                inputs=full_model.get_layer(name="image").input,
                outputs=full_model.get_layer(name='dense_1').output
            )
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Set MAX_LABEL_LENGTH (update with your actual value)
        MAX_LABEL_LENGTH = 24  # Update this based on your training data
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_image(image_path : str):
    '''
    This function loads and preprocesses images. It first receives the image path, which is used to
    decode the image as a JPEG using TensorFlow. Then, it converts the image to a tensor and applies
    two processing functions: resizing and normalization. The processed image is then returned by
    the function.

    Argument :
        image_path : The path of the image file to be loaded.

    Return:
        image : The loaded image as a tensor.
    '''

    # Read the Image
    image = tf.io.read_file(image_path)

    # Decode the image
    decoded_image = tf.image.decode_jpeg(contents = image, channels = 1)

    # Convert image data type.
    cnvt_image = tf.image.convert_image_dtype(image = decoded_image, dtype = tf.float32)

    # Resize the image
    resized_image = tf.image.resize(images = cnvt_image, size = (IMG_HEIGHT, IMG_WIDTH))

    # Transpose
    image = tf.transpose(resized_image, perm = [1, 0, 2])

    # Convert image to a tensor.
    image = tf.cast(image, dtype = tf.float32)

    # Return loaded image
    return image

def apply_craft_detection(image_path: str, output_dir: str) -> str:
    """Apply CRAFT model for text detection"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Command to run CRAFT model
        craft_command = [
            'python3', 'CRAFT_Model/CRAFT/BoundBoxFunc/test.py',
            '--cuda', '0',  # Use CPU, change to '1' if GPU available
            '--result_folder', output_dir,
            '--test_folder', os.path.dirname(image_path),
            '--trained_model', 'CRAFT_Model/CRAFT/BoundBoxFunc/weights/craft_mlt_25k.pth'
        ]
        
        # Run CRAFT detection
        result = subprocess.run(craft_command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"CRAFT detection failed: {result.stderr}")
            raise Exception(f"CRAFT detection failed: {result.stderr}")
        
        logger.info("CRAFT detection completed successfully")
        return output_dir
        
    except Exception as e:
        logger.error(f"Error in CRAFT detection: {e}")
        raise

# def sort_bounding_boxes(bounding_box_file: str) -> List[List[int]]:
#     """Sort bounding boxes based on Spanish reading order (top to bottom, left to right)"""
#     try:
#         bounding_boxes = []
        
#         with open(bounding_box_file, 'r') as f:
#             for line in f:
#                 coords = list(map(int, line.strip().split(',')[:8]))  # Take first 8 coordinates
#                 bounding_boxes.append(coords)
        
#         # Sort by y-coordinate (top to bottom), then by x-coordinate (left to right)
#         bounding_boxes.sort(key=lambda box: (box[1], box[0]))
        
#         return bounding_boxes
        
#     except Exception as e:
#         logger.error(f"Error sorting bounding boxes: {e}")
#         return []

def count_files_in_folder(folder_path, extensions_list):
    # Initialize counter for files
    file_count = 0

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file ends with the given file extension
        for extension in extensions_list:
            if filename.lower().endswith(extension):
                file_count += 1

    return file_count

def process_bounding_boxes(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse bounding box coordinates
    bounding_boxes = []
    for line in lines:
        coords = list(map(int, line.strip().split(',')))
        bounding_boxes.append(coords)

    # Sort bounding boxes based on y_min value
    bounding_boxes.sort(key=lambda box: box[1])

    vertical_distance_between_lines = 10   #Change it according to the dataset, you are using
    # Group bounding boxes based on difference between max and min y_min values
    grouped_boxes = []
    current_group = []
    for box in bounding_boxes:
        if not current_group:
            current_group.append(box)
        else:
            min_y = min(current_group, key=lambda x: x[1])[1]
            max_y = max(current_group, key=lambda x: x[1])[1]
            if box[1] - min_y <= vertical_distance_between_lines:
                current_group.append(box)
            else:
                grouped_boxes.append(current_group)
                current_group = [box]

    # Append the last group
    if current_group:
        grouped_boxes.append(current_group)

    # Sort each group based on x_min value
    for group in grouped_boxes:
        group.sort(key=lambda box: box[0])

    return grouped_boxes

def sort_bounding_boxes(bounding_box_file):
    sorted_bounding_boxes = process_bounding_boxes(bounding_box_file)

    # Write sorted bounding boxes to text file in output directory
    output_file_path = f"{os.path.splitext(bounding_box_file)[0]}_sorted.txt"
    with open(output_file_path, "w") as outfile:
        for group in sorted_bounding_boxes:
            for box in group:
                outfile.write(','.join(map(str, box)) + '\n')
            outfile.write((';'))
    return output_file_path

def extract_bounding_boxes(image_path, bounding_boxes_file, output_folder, word):
    # Read the main image
    main_image = cv2.imread(image_path)
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read bounding box coordinates from the text file
    with open(bounding_boxes_file, 'r') as f:
        bounding_boxes_data = f.read().split(';')
    bounding_boxes_data = bounding_boxes_data[1:]
    line=0
    for indx in range(len(bounding_boxes_data)-1):
        bounding_box_coords = bounding_boxes_data[indx].strip().split('\n')
        for cnt in range(len(bounding_box_coords)):
            coordinates_list = [int(coord) for coord in bounding_box_coords[cnt].split(',')]
            x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max = coordinates_list

            # Extract the bounding box from the main image
            bounding_box = main_image[y_min:y_max, x_min:x_max]

            # Save the bounding box as a separate image
            output_path = os.path.join(output_folder, f'{word};{line}.png')
            cv2.imwrite(output_path, bounding_box)

            word += 1
        line+=1

    return word

def pad_and_resize_images(folder_path):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist")

    # Define the target aspect ratio and size
    target_aspect_ratio = 4  # 1:4 aspect ratio
    target_width = 200
    target_height = 40

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                # Open the image
                with Image.open(file_path) as img:
                    img = img.convert('L')
                    width, height = img.size
                    aspect_ratio = width / height

                    if aspect_ratio < target_aspect_ratio:
                        # Calculate padding to make aspect ratio 1:4
                        new_width = height * 4
                        padding = (new_width - width) // 2
                        padded_img = ImageOps.expand(img, border=(padding, 0, padding, 0), fill='white')
                    else:
                        padded_img = img

                    # Resize the image to 200x40
                    resized_img = padded_img.resize((target_width, target_height))

                    # Save the processed image back to the original path
                    resized_img.save(file_path)

                    print(f"Processed and replaced: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

def create_csv_from_folder(folder_path, csv_file_path):
    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Create or overwrite the CSV file
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(['FILENAME', 'IDENTITY'])

        # Write data rows, excluding files with the name ".png"
        for file_name in files:
            if file_name.lower() == ".png":
                continue  # Skip files with the name ".png"

            # file_path = os.path.join(folder_path, file_name)

            # Remove the file extension (assuming it's three characters long, like '.png')
            file_name_without_extension = os.path.splitext(file_name)[0]

            csv_writer.writerow([file_name, file_name_without_extension])

    print(f'CSV file "{csv_file_path}" created successfully.')



def encode_single_sample(image_path : str, label : str):

    '''
    The function takes an image path and label as input and returns a dictionary containing the processed image tensor and the label tensor.
    First, it loads the image using the load_image function, which decodes and resizes the image to a specific size. Then it converts the given
    label string into a sequence of Unicode characters using the unicode_split function. Next, it uses the char_to_num layer to convert each
    character in the label to a numerical representation. It pads the numerical representation with a special class (n_classes)
    to ensure that all labels have the same length (MAX_LABEL_LENGTH). Finally, it returns a dictionary containing the processed image tensor
    and the label tensor.

    Arguments :
        image_path : The location of the image file.
        label      : The text to present in the image.

    Returns:
        dict : A dictionary containing the processed image and label.
    '''

    # Get the image
    image = load_image(image_path)

    # Convert the label into characters
    chars = tf.strings.unicode_split(label, input_encoding='UTF-8')

    # Convert the characters into vectors
    vecs = char_to_num(chars)

    # Pad label
    pad_size = MAX_LABEL_LENGTH - tf.shape(vecs)[0]
    vecs = tf.pad(vecs, paddings = [[0, pad_size]], constant_values=n_classes+1)

    return {'image':image, 'label':vecs}

def extract_word_images(image_path: str, bounding_boxes: List[List[int]]) -> List[np.ndarray]:
    """Extract word images from bounding boxes"""
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        word_images = []
        
        for bbox in bounding_boxes:
            # Extract coordinates
            x_coords = bbox[::2]  # Every other element starting from 0
            y_coords = bbox[1::2]  # Every other element starting from 1
            
            # Get bounding rectangle
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Extract word image
            word_img = image[y_min:y_max, x_min:x_max]
            import matplotlib.pyplot as plt
            plt.plot(word_img)
            plt.savefig("image0.png")
            if word_img.size > 0:
                word_images.append(word_img)
            
        return word_images
        
    except Exception as e:
        logger.error(f"Error extracting word images: {e}")
        return []


def preprocess_word_image(word_image: np.ndarray) -> np.ndarray:
    """
    Preprocess a word image for model input.

    Steps:
    - Convert to grayscale
    - Pad to maintain aspect ratio (target 1:4)
    - Resize to (200, 50)
    - Normalize pixel values to [0, 1]
    - Transpose shape to match (height, width, channel)

    Args:
        word_image (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: Preprocessed image ready for model input.
    """
    try:
        target_aspect_ratio = 4  # 1:4
        target_width = IMG_WIDTH
        target_height = IMG_HEIGHT

        # Convert color image to grayscale if needed
        if len(word_image.shape) == 3:
            word_image = cv2.cvtColor(word_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(word_image).convert('L')
        import matplotlib.pyplot as plt
        plt.plot(img)
        plt.savefig("image1.png")
        # Get current size and aspect ratio
        width, height = img.size
        aspect_ratio = width / height

        # Pad if aspect ratio is less than target
        if aspect_ratio < target_aspect_ratio:
            new_width = height * target_aspect_ratio
            padding = (int((new_width - width) // 2), 0)
            img = ImageOps.expand(img, border=(padding[0], 0, padding[0], 0), fill='white')

        # Resize
        img = img.resize((target_width, target_height))

        # Convert to NumPy
        img_array = np.array(img).astype(np.float32)  # shape (H, W)
        import matplotlib.pyplot as plt
        plt.plot(img)
        plt.savefig("image2.png")

        # Add channel dimension and transpose to (H, W, 1)
        img_array = img_array[:, :, np.newaxis]
        img_array = np.transpose(img_array, (1, 0, 2))  # shape (W, H, 1) → (H, W, 1)

        return img_array

    except Exception as e:
        logger.error(f"Error preprocessing word image: {e}")
        return None

def decode_prediction(pred_label: np.ndarray) -> List[str]:
    """Decode model predictions to text"""
    try:
        # Input length
        input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]
        
        # CTC decode
        decode = keras.backend.ctc_decode(
            pred_label, 
            input_length=input_len, 
            greedy=True, 
        )[0][0][:, :MAX_LABEL_LENGTH]
        
        # Convert back to characters
        chars = num_to_char(decode)
        
        # Join characters
        texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]
        
        # Clean up text
        filtered_texts = [text.replace('[UNK]', " ").strip() for text in texts]
        
        return filtered_texts
        
    except Exception as e:
        logger.error(f"Error decoding predictions: {e}")
        return []

# Set the new size in pixels (width, height) according to your choice
def resize_images_in_folder(input_folder, new_size=(200,50)):
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        # Open the image
        with Image.open(os.path.join(input_folder, filename)) as img:
            # Resize the image
            resized_img = img.resize(new_size)
            # Save the resized image to the output folder
            output_filename = os.path.splitext(filename)[0] + '.png'  # Ensure output format is PNG
            resized_img.save(os.path.join(input_folder, output_filename))

def decode_pred(pred_label):

    '''
    The decode_pred function is used to decode the predicted labels generated by the OCR model.
    It takes a matrix of predicted labels as input, where each time step represents the probability
    for each character. The function uses CTC decoding to decode the numeric labels back into their
    character values. The function also removes any unknown tokens and returns the decoded texts as a
    list of strings. The function utilizes the num_to_char function to map numeric values back to their
    corresponding characters. Overall, the function is an essential step in the OCR process, as it allows
    us to obtain the final text output from the model's predictions.

    Argument :
        pred_label : These are the model predictions which are needed to be decoded.

    Return:
        filtered_text : This is the list of all the decoded and processed predictions.

    '''

    # Input length
    input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]

    # CTC decode
    decode = keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=False, beam_width=5)[0][0][:,:MAX_LABEL_LENGTH]

    # Converting numerics back to their character values
    chars = num_to_char(decode)

    # Join all the characters
    texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]

    # Remove the unknown token
    filtered_texts = [text.replace('[UNK]', " ").strip() for text in texts]

    return filtered_texts

def predict_word_images(word_images: List[np.ndarray]) -> List[str]:
    """Predict text from word images"""
    try:
        if not word_images:
            return []
        
        # Preprocess all word images
        processed_images = []
        for word_img in word_images:
            processed_img = preprocess_word_image(word_img)
            if processed_img is not None:
                processed_images.append(processed_img)
        
        if not processed_images:
            return []
        
        # Batch predict
        batch_images = np.array(processed_images)
        predictions = inference_model.predict(batch_images)
        
        # Decode predictions
        decoded_texts = decode_prediction(predictions)
        
        # Apply character corrections
        corrected_texts = [replace_chars(text) for text in decoded_texts]
        
        return corrected_texts
        
    except Exception as e:
        logger.error(f"Error predicting word images: {e}")
        return []

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        load_model_and_setup()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

@app.post("/ocr/predict", response_model=Dict[str, Any])
async def predict_text(file: UploadFile = File(...)):
    """
    Extract and recognize text from an uploaded image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded image
        contents = await file.read()
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded image
            image_path = os.path.join(temp_dir, f"input_{file.filename}")
            with open(image_path, 'wb') as f:
                f.write(contents)
            
            # Apply CRAFT detection
            craft_output_dir = os.path.join(temp_dir, "craft_output/")
            craft_output_dir = apply_craft_detection(image_path, craft_output_dir)
            # Find bounding box file
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            bbox_file = os.path.join(craft_output_dir, f"res_{image_basename}.txt")
            if not os.path.exists(bbox_file):
                raise HTTPException(status_code=404, detail="No text detected in image")
            
            # Sort bounding boxes
            sorted_file = sort_bounding_boxes(bbox_file)
            if not sorted_file:
                raise HTTPException(status_code=404, detail="No valid bounding boxes found")
            
            # Extract word images
            word = 0
            output_folder = os.path.join(temp_dir, "extracted_word_images/")
            word = extract_bounding_boxes(image_path, sorted_file, output_folder, word)
            pad_and_resize_images(output_folder)
            test_csv_path = os.path.join(temp_dir, 'testing_data.csv')
            create_csv_from_folder(output_folder, test_csv_path)

            test_csv = pd.read_csv(test_csv_path)
            test_csv['IDENTITY'] = test_csv['IDENTITY'].apply(lambda x: str(x))
            test_csv['FILENAME']  = [output_folder + f"/{filename}" for filename in test_csv['FILENAME']]

            resize_images_in_folder(output_folder)

            df_infer = test_csv

            # Step 1: Sort the dataframe based on values before ';'
            df_infer['before_semicolon'] = df_infer['IDENTITY'].apply(lambda x: int(x.split(';')[0]))
            df_infer['after_semicolon'] = df_infer['IDENTITY'].apply(lambda x: int(x.split(';')[1]))
            sorted_df = df_infer.sort_values(['before_semicolon']).reset_index(drop=True)
            sorted_df.drop(columns=['before_semicolon', 'after_semicolon'], inplace=True)

            sorted_df['IDENTITY'] = sorted_df['IDENTITY'].astype(str)

            sorted_dfs = tf.data.Dataset.from_tensor_slices(
                (np.array(sorted_df['FILENAME'].to_list()), np.array(sorted_df['IDENTITY'].to_list()))
            ).map(encode_single_sample, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

            decoded_predictions = decode_pred(inference_model.predict(sorted_dfs))
            pred = sorted_df['IDENTITY'].tolist()
            formatted_output = []

            current_group = None
            i = 0
            for prediction in pred:
                before, after = map(int, prediction.split(';'))

                if current_group is None:
                    current_group = after

                if after != current_group:
                    formatted_output.append('\n')  # Start a new line for the new group
                    current_group = after

                formatted_output.append(decoded_predictions[i] + ' ')
                i += 1

            formatted_output.append('\n')  # Final new line

            full_text = ''.join(formatted_output)


            # if not output_folder:
            #     raise HTTPException(status_code=404, detail="No word images extracted")
            

            # # Predict text from word images
            # predictions = predict_word_images(word_images)
            # # Combine predictions into full text
            # full_text = ' '.join(predictions)
            print(full_text)
            return {
                "status": "success",
                "extracted_text": full_text,
                # "word_count": len(predictions),
                # "words": predictions,
                "message": "Text extracted successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": inference_model is not None}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Spanish OCR API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/ocr/predict",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)