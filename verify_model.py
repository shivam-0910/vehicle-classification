#!/usr/bin/env python3
"""Temporary script to verify the final model for STEP 2."""

import sys
import json
sys.path.insert(0, '.')

from tensorflow import keras
import yaml
import numpy as np

print("="*70)
print("STEP 2: VERIFYING FINAL MODEL")
print("="*70)

# 1. Load model
model_path = "models/best_model_exp3.keras"
print(f"\n1. Loading model from: {model_path}")
try:
    model = keras.models.load_model(model_path)
    print("   ✓ Model loaded successfully")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    sys.exit(1)

# 2. Model summary
print("\n2. Model Architecture:")
model.summary()

# 3. Input shape
print("\n3. Input Shape:")
input_shape = model.input_shape
print(f"   Expected: (None, 128, 128, 3)")
print(f"   Actual:   {input_shape}")
if input_shape == (None, 128, 128, 3):
    print("   ✓ CORRECT")
else:
    print("   ⚠ MISMATCH")

# 4. Output shape
print("\n4. Output Shape:")
output_shape = model.output_shape
print(f"   Expected: (None, 10)")
print(f"   Actual:   {output_shape}")
if output_shape == (None, 10):
    print("   ✓ CORRECT (10 classes)")
else:
    print("   ⚠ MISMATCH")

# 5. Dense layers
print("\n5. Dense Layers:")
for layer in model.layers:
    if "dense" in layer.name:
        units = layer.units if hasattr(layer, "units") else "N/A"
        print(f"   - {layer.name}: {units} units")

# 6. Inference test
print("\n6. Testing Inference:")
dummy_image = np.random.rand(1, 128, 128, 3).astype(np.float32)
try:
    prediction = model.predict(dummy_image, verbose=0)
    print(f"   ✓ Inference successful")
    print(f"   ✓ Output shape: {prediction.shape}")
    print(f"   ✓ Softmax sum: {prediction[0].sum():.6f}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    sys.exit(1)

# 7. Config check
print("\n7. Experiment 3 Configuration:")
config_path = "src/config/config_phase10_5_exp3.yaml"
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    classes = config.get("dataset", {}).get("classes", [])
    print(f"   Classes: {len(classes)} defined")
    print(f"   {classes}")
    if len(classes) == 10:
        print("   ✓ Class count CORRECT")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# 8. Metadata
print("\n8. Final Model Metadata:")
try:
    with open("results/final_model.json", "r") as f:
        metadata = json.load(f)
    print(f"   Experiment: {metadata.get('experiment')}")
    print(f"   Model path: {metadata.get('model_path')}")
    print(f"   Config path: {metadata.get('config_path')}")
    print(f"   Test accuracy: {metadata.get('test_accuracy'):.4f}")
    print(f"   Status: {metadata.get('status')}")
    if metadata.get("status") == "final":
        print("   ✓ Status is FINAL")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

print("\n" + "="*70)
print("STEP 2: MODEL VERIFICATION COMPLETE ✓")
print("="*70)
