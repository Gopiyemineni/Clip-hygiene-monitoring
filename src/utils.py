
import os
import logging
from PIL import Image
import requests
from io import BytesIO

def load_image(image_path):
    """Load image from file path or URL"""
    logging.info(f"Loading image from {image_path}...")
    try:
        if image_path.startswith('http'):
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        logging.info("✓ Image loaded successfully.")
        return image
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

def print_assessment_results(result, image_path=""):
    """Print formatted assessment results"""
    if 'error' in result:
        logging.error(f"✗ Assessment failed for {image_path}")
        return

    result_str = f"\n{'='*60}\n"
    result_str += "HYGIENE ASSESSMENT RESULTS\n"
    if image_path:
        result_str += f"Image: {os.path.basename(image_path)}\n"
    result_str += f"{'='*60}\n"
    result_str += f"Area Type: {result['area_type'].upper()}\n"
    result_str += f"Classification: {result['classification']}\n"
    result_str += f"Confidence: {result['confidence']:.2%}\n"
    result_str += "\nOverall Scores:\n"
    result_str += f"  Hygienic: {result['overall_scores']['hygienic']:.3f}\n"
    result_str += f"  Unhygienic: {result['overall_scores']['unhygienic']:.3f}\n"
    result_str += "\nTop Indicators:\n"
    result_str += "  Best hygienic match:\n"
    result_str += f"    '{result['top_indicators']['most_hygienic_match']['description']}'\n"
    result_str += f"    Score: {result['top_indicators']['most_hygienic_match']['score']:.3f}\n"
    result_str += "  Best unhygienic match:\n"
    result_str += f"    '{result['top_indicators']['most_unhygienic_match']['description']}'\n"
    result_str += f"    Score: {result['top_indicators']['most_unhygienic_match']['score']:.3f}"

    logging.info(result_str)
