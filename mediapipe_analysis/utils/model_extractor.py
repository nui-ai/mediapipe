#!/usr/bin/env python3
"""
Model extractor for MediaPipe TensorFlow Lite models.
This module demonstrates how to extract the neural network models
used by MediaPipe from the installed package.

Author: GitHub Copilot
"""

import shutil
import os
from pathlib import Path
from typing import List, Tuple


def find_mediapipe_models() -> List[Tuple[str, Path]]:
    """
    Find all TensorFlow Lite models in the installed MediaPipe package.
    
    Returns:
        List of (model_name, model_path) tuples
    """
    try:
        import mediapipe
        mediapipe_root = Path(mediapipe.__file__).parent
    except ImportError:
        print("MediaPipe not installed")
        return []
    
    models = []
    for tflite_file in mediapipe_root.rglob("*.tflite"):
        # Get relative name for better organization
        relative_path = tflite_file.relative_to(mediapipe_root)
        model_name = str(relative_path).replace('/', '_').replace('\\', '_')
        models.append((model_name, tflite_file))
    
    return models


def extract_hand_models(target_dir: Path) -> bool:
    """
    Extract the specific hand inference models to target directory.
    
    Args:
        target_dir: Directory to extract models to
        
    Returns:
        True if extraction successful, False otherwise
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    models = find_mediapipe_models()
    hand_models = []
    
    for model_name, model_path in models:
        if 'palm_detection_full.tflite' in str(model_path):
            hand_models.append(('palm_detection_full.tflite', model_path))
        elif 'hand_landmark_full.tflite' in str(model_path):
            hand_models.append(('hand_landmark_full.tflite', model_path))
    
    if len(hand_models) != 2:
        print(f"Expected 2 hand models, found {len(hand_models)}")
        for name, path in hand_models:
            print(f"  {name}: {path}")
        return False
    
    # Extract models
    for target_name, source_path in hand_models:
        target_path = target_dir / target_name
        shutil.copy2(source_path, target_path)
        print(f"✓ Extracted {target_name} ({source_path.stat().st_size} bytes)")
        print(f"  From: {source_path}")
        print(f"  To: {target_path}")
    
    return True


def verify_extracted_models(models_dir: Path) -> bool:
    """
    Verify that extracted models are valid and loadable.
    
    Args:
        models_dir: Directory containing extracted models
        
    Returns:
        True if all models are valid
    """
    required_models = [
        'palm_detection_full.tflite',
        'hand_landmark_full.tflite'
    ]
    
    for model_name in required_models:
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"✗ Missing model: {model_path}")
            return False
        
        # Try to load with TensorFlow Lite
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"✓ {model_name} is valid")
            print(f"  Input shape: {input_details[0]['shape']}")
            print(f"  Outputs: {len(output_details)}")
            
        except Exception as e:
            print(f"✗ {model_name} failed to load: {e}")
            return False
    
    return True


if __name__ == "__main__":
    print("MediaPipe Model Extractor")
    print("=" * 40)
    
    # Show all available models
    print("\nAvailable MediaPipe models:")
    models = find_mediapipe_models()
    for model_name, model_path in models:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  {model_name}: {size_mb:.1f} MB")
    
    # Extract hand models to repository
    repo_root = Path(__file__).parent.parent.parent.parent
    target_dir = repo_root / "tflite"
    
    print(f"\nExtracting hand models to: {target_dir}")
    if extract_hand_models(target_dir):
        print("\n✓ Hand models extracted successfully")
        
        print("\nVerifying extracted models:")
        if verify_extracted_models(target_dir):
            print("✓ All models verified and ready for use")
        else:
            print("✗ Some models failed verification")
    else:
        print("✗ Failed to extract hand models")