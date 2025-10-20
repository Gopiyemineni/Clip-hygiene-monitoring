
import sys
from src.hygiene_classifier import HygieneClassifier, analyze_hygiene_from_path, analyze_multiple_images, interactive_demo

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
