
import unittest
from unittest.mock import MagicMock, patch
from src.hygiene_classifier import HygieneClassifier
from PIL import Image
import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding

class TestHygieneClassifier(unittest.TestCase):

    @patch('src.hygiene_classifier.CLIPProcessor.from_pretrained')
    @patch('src.hygiene_classifier.CLIPModel.from_pretrained')
    def setUp(self, mock_model_from_pretrained, mock_processor_from_pretrained):
        # Mock for CLIPModel
        self.mock_model_instance = MagicMock()
        self.mock_model_instance.to.return_value = self.mock_model_instance
        mock_model_from_pretrained.return_value = self.mock_model_instance

        # Mock for CLIPProcessor
        self.mock_processor_instance = MagicMock()
        mock_processor_from_pretrained.return_value = self.mock_processor_instance

        # Mock the __call__ method of the processor to return a BatchEncoding object
        processor_output_dict = {
            'input_ids': torch.randint(0, 100, (1, 77)),
            'pixel_values': torch.randn(1, 3, 224, 224),
            'attention_mask': torch.ones(1, 77)
        }
        self.mock_processor_instance.return_value = BatchEncoding(processor_output_dict)

        self.classifier = HygieneClassifier(config_path='config.json')

    def test_classify_area_type(self):
        image = Image.new('RGB', (224, 224))

        # Mock the model's output for this test
        mock_output = MagicMock()
        mock_output.logits_per_image = torch.tensor([[1.0, 5.0, 1.0, 1.0]])
        self.mock_model_instance.return_value = mock_output

        area_type, confidence = self.classifier.classify_area_type(image)

        self.assertEqual(area_type, 'kitchen')
        self.assertGreater(confidence, 0.5)

    def test_assess_hygiene(self):
        image = Image.new('RGB', (224, 224))

        # Mock the model's output for area classification
        mock_area_output = MagicMock()
        mock_area_output.logits_per_image = torch.tensor([[1.0, 5.0, 1.0, 1.0]])

        # Mock the model's output for hygiene assessment
        mock_hygiene_output = MagicMock()
        hygienic_logits = [1.0] * 5
        unhygienic_logits = [2.0] * 5
        mock_hygiene_output.logits_per_image = torch.tensor([hygienic_logits + unhygienic_logits])

        self.mock_model_instance.side_effect = [
            mock_area_output,
            mock_hygiene_output
        ]

        classification, confidence, details = self.classifier.assess_hygiene(image)

        self.assertEqual(classification, 'Unhygienic')
        self.assertGreater(confidence, 0.5)

if __name__ == '__main__':
    unittest.main()
