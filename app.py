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

class HygieneClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """Initialize CLIP model using transformers library"""
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            print(f"✓ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Make sure you have internet connection for first-time download.")
            raise
        
        # Define hygiene indicators for different areas
        self.hygiene_prompts = {
            'toilet': {
                'hygienic': [
                    "a clean toilet with no stains or dirt",
                    "a spotless bathroom with clean tiles",
                    "a well-maintained toilet area",
                    "a sanitized restroom facility",
                    "a pristine toilet bowl and surrounding area"
                ],
                'unhygienic': [
                    "a dirty toilet with stains and grime",
                    "an unclean bathroom with visible dirt",
                    "a neglected toilet area with poor maintenance",
                    "an unsanitary restroom facility",
                    "a filthy toilet with visible contamination"
                ]
            },
            'kitchen': {
                'hygienic': [
                    "a clean kitchen with spotless countertops",
                    "a well-organized kitchen with clean surfaces",
                    "a sanitized cooking area",
                    "a pristine kitchen with clean appliances",
                    "a hygienic food preparation area"
                ],
                'unhygienic': [
                    "a dirty kitchen with food debris and stains",
                    "an unclean cooking area with grease buildup",
                    "a messy kitchen with unwashed dishes",
                    "a contaminated food preparation surface",
                    "a filthy kitchen with poor sanitation"
                ]
            },
            'hospital': {
                'hygienic': [
                    "a sterile hospital room with clean equipment",
                    "a sanitized medical facility",
                    "a pristine healthcare environment",
                    "a well-maintained hospital area",
                    "a hygienic medical treatment room"
                ],
                'unhygienic': [
                    "an unclean hospital room with contamination",
                    "a poorly maintained medical facility",
                    "a dirty healthcare environment",
                    "an unsanitary hospital area",
                    "a contaminated medical treatment space"
                ]
            },
            'dining': {
                'hygienic': [
                    "a clean dining table with spotless surface",
                    "a well-set dining area with clean tableware",
                    "a sanitized eating area",
                    "a pristine dining table ready for meals",
                    "a hygienic food serving area"
                ],
                'unhygienic': [
                    "a dirty dining table with food remnants",
                    "an unclean eating area with stains",
                    "a messy dining table with spills",
                    "a contaminated food serving surface",
                    "a filthy dining area with poor cleanliness"
                ]
            }
        }
    
    def classify_area_type(self, image):
        """Classify what type of area the image shows"""
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
            return area_types[max_idx], probs[max_idx]
            
        except Exception as e:
            print(f"Error in area classification: {e}")
            return "unknown", 0.0
    
    def assess_hygiene(self, image, area_type=None):
        """Assess hygiene level of the given image"""
        
        # First classify area type if not provided
        if area_type is None:
            area_type, confidence = self.classify_area_type(image)
            print(f"Detected area type: {area_type} (confidence: {confidence:.3f})")
        
        # Get relevant prompts for the area type
        if area_type not in self.hygiene_prompts:
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
            print(f"Error in hygiene assessment: {e}")
            return "Error", 0.0, {}
    
    def detailed_assessment(self, image, area_type=None):
        """Provide detailed hygiene assessment with specific indicators"""
        classification, confidence, details = self.assess_hygiene(image, area_type)
        
        if not details:
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
        
        return result
    
    def batch_assess_hygiene(self, images, area_types=None):
        """Assess hygiene for multiple images at once"""
        results = []
        for i, image in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}...")
            area_type = area_types[i] if area_types and i < len(area_types) else None
            result = self.detailed_assessment(image, area_type)
            results.append(result)
        return results

def load_image(image_path):
    """Load image from file path or URL"""
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
        
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def print_assessment_results(result, image_path=""):
    """Print formatted assessment results"""
    if 'error' in result:
        print(f"✗ Assessment failed for {image_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"HYGIENE ASSESSMENT RESULTS")
    if image_path:
        print(f"Image: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    print(f"Area Type: {result['area_type'].upper()}")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nOverall Scores:")
    print(f"  Hygienic: {result['overall_scores']['hygienic']:.3f}")
    print(f"  Unhygienic: {result['overall_scores']['unhygienic']:.3f}")
    print(f"\nTop Indicators:")
    print(f"  Best hygienic match:")
    print(f"    '{result['top_indicators']['most_hygienic_match']['description']}'")
    print(f"    Score: {result['top_indicators']['most_hygienic_match']['score']:.3f}")
    print(f"  Best unhygienic match:")
    print(f"    '{result['top_indicators']['most_unhygienic_match']['description']}'")
    print(f"    Score: {result['top_indicators']['most_unhygienic_match']['score']:.3f}")

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
            print(f"Skipping {path} due to loading error")
    
    if not images:
        print("No valid images to process")
        return []
    
    results = classifier.batch_assess_hygiene(images, area_types)
    
    # Print results
    for i, result in enumerate(results):
        print_assessment_results(result, valid_paths[i])
    
    return results

def interactive_demo():
    """Interactive demonstration of the hygiene classifier"""
    print("\n" + "="*60)
    print("CLIP HYGIENE CLASSIFIER - INTERACTIVE DEMO")
    print("="*60)
    
    # Initialize classifier
    try:
        classifier = HygieneClassifier()
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
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
            
            print("Testing with demo images...")
            for area, url in demo_images.items():
                print(f"\nTesting {area} image...")
                try:
                    analyze_hygiene_from_path(url, area, classifier)
                except Exception as e:
                    print(f"Error with demo image: {e}")
        
        elif choice == '4':
            print("\nSupported area types:")
            for area in classifier.hygiene_prompts.keys():
                print(f"  - {area}")
        
        elif choice == '5':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1-5.")

def main():
    """Main function"""
    print("🧹 CLIP Hygiene Classification System")
    print("=====================================")
    
    if len(sys.argv) > 1:
        # Command line usage
        image_paths = sys.argv[1:]
        print(f"Analyzing {len(image_paths)} image(s)...")
        
        if len(image_paths) == 1:
            analyze_hygiene_from_path(image_paths[0])
        else:
            analyze_multiple_images(image_paths)
    else:
        # Interactive mode
        interactive_demo()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()