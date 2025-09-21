#!/usr/bin/env python3
"""
TensorFlow Lite model inspector for MediaPipe hand inference models.
Analyzes the palm detection and hand landmark models to understand their
input/output specifications and preprocessing requirements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import struct
import numpy as np
from pathlib import Path

# Try to import TensorFlow, fallback to basic analysis if not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available, using basic model analysis")
    TF_AVAILABLE = False


class TFLiteModelInspector:
    """Inspector for TensorFlow Lite models used in MediaPipe hand inference."""
    
    def __init__(self, model_path: str):
        """Initialize with path to TFLite model."""
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        if TF_AVAILABLE:
            self._load_model()
        else:
            self._basic_model_info()
    
    def _load_model(self):
        """Load the TFLite model and get input/output details."""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            print(f"Error loading model {self.model_path}: {e}")
            raise
    
    def _basic_model_info(self):
        """Get basic model info without TensorFlow."""
        model_name = Path(self.model_path).name
        file_size = os.path.getsize(self.model_path)
        
        # Basic file analysis
        print(f"Model: {model_name}")
        print(f"File size: {file_size:,} bytes")
        
        # Try to read TFLite file header
        try:
            with open(self.model_path, 'rb') as f:
                # TFLite files start with "TFL3" magic number
                magic = f.read(8)
                if magic[:4] == b'TFL3':
                    print("✓ Valid TensorFlow Lite model file")
                else:
                    print("⚠ Not a standard TFLite file format")
        except Exception as e:
            print(f"Could not read file: {e}")
    
    def inspect(self):
        """Inspect and print model details."""
        model_name = Path(self.model_path).name
        print(f"\n=== Model: {model_name} ===")
        
        if not TF_AVAILABLE:
            self._basic_model_info()
            return {'model_name': model_name, 'input_details': [], 'output_details': []}
        
        print(f"\nInput Details:")
        for i, input_detail in enumerate(self.input_details):
            print(f"  Input {i}:")
            print(f"    Name: {input_detail['name']}")
            print(f"    Shape: {input_detail['shape']}")
            print(f"    Type: {input_detail['dtype']}")
            print(f"    Quantization: {input_detail.get('quantization', 'None')}")
        
        print(f"\nOutput Details:")
        for i, output_detail in enumerate(self.output_details):
            print(f"  Output {i}:")
            print(f"    Name: {output_detail['name']}")
            print(f"    Shape: {output_detail['shape']}")
            print(f"    Type: {output_detail['dtype']}")
            print(f"    Quantization: {output_detail.get('quantization', 'None')}")
        
        return {
            'model_name': model_name,
            'input_details': self.input_details,
            'output_details': self.output_details
        }
    
    def test_inference(self, dummy_input_shape=None):
        """Test inference with dummy input to verify model works."""
        if not TF_AVAILABLE:
            print("Cannot test inference without TensorFlow")
            return None
            
        try:
            if dummy_input_shape is None:
                dummy_input_shape = self.input_details[0]['shape']
            
            # Create dummy input
            dummy_input = np.random.rand(*dummy_input_shape).astype(self.input_details[0]['dtype'])
            
            # Set input
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get outputs
            outputs = []
            for output_detail in self.output_details:
                output = self.interpreter.get_tensor(output_detail['index'])
                outputs.append(output)
                print(f"Output {output_detail['name']}: shape {output.shape}, dtype {output.dtype}")
            
            print("✓ Model inference test successful")
            return outputs
            
        except Exception as e:
            print(f"✗ Model inference test failed: {e}")
            return None


def inspect_mediapipe_models():
    """Inspect both MediaPipe hand inference models."""
    base_path = Path(__file__).parent.parent  # Go up to repo root
    tflite_path = base_path / "models"
    
    models = [
        "palm_detection_full.tflite",
        "hand_landmark_full.tflite"
    ]
    
    results = {}
    
    for model_name in models:
        model_path = tflite_path / model_name
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        
        try:
            inspector = TFLiteModelInspector(str(model_path))
            model_info = inspector.inspect()
            inspector.test_inference()
            results[model_name] = model_info
        except Exception as e:
            print(f"Failed to inspect {model_name}: {e}")
    
    return results


if __name__ == "__main__":
    print("MediaPipe Hand Inference Model Inspector")
    print("=" * 50)
    
    results = inspect_mediapipe_models()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for model_name, info in results.items():
        print(f"\n{model_name}:")
        if info['input_details'] and len(info['input_details']) > 0:
            input_shape = info['input_details'][0]['shape']
            input_type = info['input_details'][0]['dtype']
            print(f"  Input: {input_shape} ({input_type})")
        else:
            print(f"  Input: Unable to determine (TensorFlow not available)")
        
        print(f"  Outputs: {len(info['output_details'])}")
        for i, output in enumerate(info['output_details']):
            print(f"    {i}: {output['shape']} ({output['dtype']})")
            
    if not TF_AVAILABLE:
        print(f"\nNote: Install TensorFlow for detailed model analysis:")
        print(f"pip install tensorflow")