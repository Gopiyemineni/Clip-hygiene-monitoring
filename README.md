🧹 CLIP Hygiene Monitoring

AI-powered hygiene monitoring system using OpenAI’s CLIP model.
This project classifies images from hospitals, toilets, kitchens, and dining areas as Hygienic or Unhygienic and provides detailed analysis with confidence scores and top matching indicators.

✨ Features

🔍 Automatic area detection (toilet, kitchen, hospital, dining table)

✅ Hygienic vs. Unhygienic classification

📊 Confidence scores and detailed hygiene indicators

🖼️ Supports local images and URLs

🧾 Batch analysis for multiple images

🎛️ Interactive CLI demo

📦 Installation

Clone this repository and install the required dependencies:

git clone https://github.com/Gopiyemineni/Clip-hygiene-monitoring.git
cd Clip-hygiene-monitoring
pip install -r requirements.txt


Or install dependencies manually:

pip install transformers torch torchvision pillow requests numpy

🚀 Usage
1️⃣ Run Interactive Demo
python hygiene_classifier.py

2️⃣ Analyze a Single Image
python hygiene_classifier.py path/to/image.jpg

3️⃣ Analyze Multiple Images
python hygiene_classifier.py image1.jpg image2.jpg image3.jpg

4️⃣ Analyze from URL
from hygiene_classifier import analyze_hygiene_from_path

result = analyze_hygiene_from_path("https://images.unsplash.com/photo-1620626011761-996317b8d101?w=500")
print(result)
