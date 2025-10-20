"""
Complete CLIP Hygiene Classification System
==========================================

This system uses CLIP transformer model to classify hygiene levels in different areas:
- Toilets/Bathrooms
- Kitchens
- Hospital areas
- Dining tables

Installation:
pip install transformers torch torchvision pillow requests numpy

Usage:
python hygiene_classifier.py
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import os
import sys
import json
import logging
from .utils import load_image, print_assessment_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HygieneClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32", config_path="config.json"):
        """Initialize CLIP model using transformers library"""
        logging.info("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            logging.info(f"✓ Model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"✗ Error loading model: {e}")
            logging.error("Make sure you have internet connection for first-time download.")
            raise

        # Load hygiene indicators from config file
        self.hygiene_prompts = self._load_config(config_path)

    def _load_config(self, config_path):
        """Load hygiene prompts from a JSON config file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"✓ Config loaded successfully from {config_path}")
            return config['hygiene_prompts']
        except FileNotFoundError:
            logging.error(f"✗ Error: Config file not found at {config_path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"✗ Error: Invalid JSON in config file {config_path}")
            raise
        except KeyError:
            logging.error("✗ Error: 'hygiene_prompts' key not found in config file")
            raise

    def classify_area_type(self, image):
        """Classify what type of area the image shows"""
        logging.info("Classifying area type...")
        area_prompts = [
            "a toilet or bathroom",
            "a kitchen or cooking area",
            "a hospital or medical facility",
            "a dining table or eating area"
        ]

        area_types = ['toilet', 'kitchen', 'hospital', 'dining']

        try:
            # Process inputs
            inputs = self.processor(
                text=area_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            # Return the area type with highest probability
            max_idx = np.argmax(probs)
            area_type = area_types[max_idx]
            confidence = probs[max_idx]
            logging.info(f"✓ Area classified as '{area_type}' with confidence {confidence:.3f}")
            return area_type, confidence

        except Exception as e:
            logging.error(f"Error in area classification: {e}")
            return "unknown", 0.0

    def assess_hygiene(self, image, area_type=None):
        """Assess hygiene level of the given image"""
        logging.info("Assessing hygiene...")

        # First classify area type if not provided
        if area_type is None:
            area_type, confidence = self.classify_area_type(image)

        # Get relevant prompts for the area type
        if area_type not in self.hygiene_prompts:
            logging.error(f"Unknown area type: {area_type}")
            return "Unknown area type", 0.0, {}

        prompts = self.hygiene_prompts[area_type]
        all_prompts = prompts['hygienic'] + prompts['unhygienic']
        hygienic_count = len(prompts['hygienic'])

        try:
            # Process inputs
            inputs = self.processor(
                text=all_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            # Calculate hygiene scores
            hygienic_score = np.mean(probs[:hygienic_count])
            unhygienic_score = np.mean(probs[hygienic_count:])

            # Determine classification
            if hygienic_score > unhygienic_score:
                classification = "Hygienic"
                confidence = hygienic_score / (hygienic_score + unhygienic_score)
            else:
                classification = "Unhygienic"
                confidence = unhygienic_score / (hygienic_score + unhygienic_score)

            logging.info(f"✓ Hygiene assessed as '{classification}' with confidence {confidence:.2%}")

            return classification, confidence, {
                'area_type': area_type,
                'hygienic_score': float(hygienic_score),
                'unhygienic_score': float(unhygienic_score),
                'detailed_scores': {
                    'hygienic_prompts': [float(score) for score in probs[:hygienic_count]],
                    'unhygienic_prompts': [float(score) for score in probs[hygienic_count:]]
                }
            }

        except Exception as e:
            logging.error(f"Error in hygiene assessment: {e}")
            return "Error", 0.0, {}

    def detailed_assessment(self, image, area_type=None):
        """Provide detailed hygiene assessment with specific indicators"""
        logging.info("Performing detailed assessment...")
        classification, confidence, details = self.assess_hygiene(image, area_type)

        if not details:
            logging.error("Assessment failed, no details available.")
            return {'error': 'Assessment failed'}

        area_type = details['area_type']

        # Get the most relevant indicators
        hygienic_scores = details['detailed_scores']['hygienic_prompts']
        unhygienic_scores = details['detailed_scores']['unhygienic_prompts']

        hygienic_prompts = self.hygiene_prompts[area_type]['hygienic']
        unhygienic_prompts = self.hygiene_prompts[area_type]['unhygienic']

        # Find top indicators
        top_hygienic_idx = np.argmax(hygienic_scores)
        top_unhygienic_idx = np.argmax(unhygienic_scores)

        result = {
            'classification': classification,
            'confidence': confidence,
            'area_type': area_type,
            'overall_scores': {
                'hygienic': details['hygienic_score'],
                'unhygienic': details['unhygienic_score']
            },
            'top_indicators': {
                'most_hygienic_match': {
                    'description': hygienic_prompts[top_hygienic_idx],
                    'score': hygienic_scores[top_hygienic_idx]
                },
                'most_unhygienic_match': {
                    'description': unhygienic_prompts[top_unhygienic_idx],
                    'score': unhygienic_scores[top_unhygienic_idx]
                }
            }
        }

        logging.info("✓ Detailed assessment complete.")
        return result

    def batch_assess_hygiene(self, images, area_types=None):
        """Assess hygiene for multiple images at once"""
        results = []
        for i, image in enumerate(images):
            logging.info(f"Processing image {i+1}/{len(images)}...")
            area_type = area_types[i] if area_types and i < len(area_types) else None
            result = self.detailed_assessment(image, area_type)
            results.append(result)
        return results

def analyze_hygiene_from_path(image_path, area_type=None, classifier=None):
    """Analyze hygiene from image file path"""
    if classifier is None:
        classifier = HygieneClassifier()

    # Load image
    image = load_image(image_path)
    if image is None:
        return None

    # Get detailed assessment
    result = classifier.detailed_assessment(image, area_type)

    # Print results
    print_assessment_results(result, image_path)

    return result

def analyze_multiple_images(image_paths, area_types=None, classifier=None):
    """Analyze multiple images for hygiene"""
    if classifier is None:
        classifier = HygieneClassifier()

    images = []
    valid_paths = []

    for path in image_paths:
        image = load_image(path)
        if image is not None:
            images.append(image)
            valid_paths.append(path)
        else:
            logging.warning(f"Skipping {path} due to loading error")

    if not images:
        logging.warning("No valid images to process")
        return []

    results = classifier.batch_assess_hygiene(images, area_types)

    # Print results
    for i, result in enumerate(results):
        print_assessment_results(result, valid_paths[i])

    return results

def interactive_demo():
    """Interactive demonstration of the hygiene classifier"""
    logging.info("Starting interactive demo...")
    print("\n" + "="*60)
    print("CLIP HYGIENE CLASSIFIER - INTERACTIVE DEMO")
    print("="*60)

    # Initialize classifier
    try:
        classifier = HygieneClassifier()
    except Exception as e:
        logging.error(f"Failed to initialize classifier: {e}")
        return

    while True:
        print("\nOptions:")
        print("1. Analyze single image")
        print("2. Analyze multiple images")
        print("3. Test with online demo images")
        print("4. Show supported area types")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == '1':
            image_path = input("Enter image path or URL: ").strip()
            if image_path:
                area_type = input("Specify area type (toilet/kitchen/hospital/dining) or press Enter for auto-detection: ").strip()
                area_type = area_type if area_type in ['toilet', 'kitchen', 'hospital', 'dining'] else None
                analyze_hygiene_from_path(image_path, area_type, classifier)

        elif choice == '2':
            print("Enter image paths (one per line, empty line to finish):")
            image_paths = []
            while True:
                path = input(f"Image {len(image_paths) + 1}: ").strip()
                if not path:
                    break
                image_paths.append(path)

            if image_paths:
                analyze_multiple_images(image_paths, classifier=classifier)

        elif choice == '3':
            demo_images = {
                'kitchen': 'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=500',
                'bathroom': 'https://images.unsplash.com/photo-1620626011761-996317b8d101?w=500'
            }

            logging.info("Testing with demo images...")
            for area, url in demo_images.items():
                logging.info(f"\nTesting {area} image...")
                try:
                    analyze_hygiene_from_path(url, area, classifier)
                except Exception as e:
                    logging.error(f"Error with demo image: {e}")

        elif choice == '4':
            print("\nSupported area types:")
            for area in classifier.hygiene_prompts.keys():
                print(f"  - {area}")

        elif choice == '5':
            logging.info("Exiting interactive demo.")
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please select 1-5.")
