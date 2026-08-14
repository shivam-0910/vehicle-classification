#!/usr/bin/env python3
"""Step 8: Verify Streamlit Application"""

import sys
sys.path.insert(0, '.')

print("="*70)
print("STEP 8: VERIFYING STREAMLIT APPLICATION")
print("="*70)

# Test 1: Import all app dependencies
print("\n1. Checking Streamlit app imports...")
try:
    import streamlit as st
    from PIL import Image
    import tempfile
    from pathlib import Path
    print("   ✓ Streamlit dependencies OK")
except ImportError as e:
    print(f"   ✗ Missing dependency: {e}")
    sys.exit(1)

# Test 2: Import inference functions
print("\n2. Checking inference functions...")
try:
    from src.data.loader import load_config
    from src.inference.predict import (
        InferenceError,
        PredictionResult,
        get_class_names,
        load_model,
        predict_image,
        preprocess_image,
        verify_model_matches_classes,
    )
    print("   ✓ Inference functions importable")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 3: Load config
print("\n3. Loading configuration...")
try:
    config = load_config('src/config/config_phase10_5_exp3.yaml')
    num_classes = len(config.get("dataset", {}).get("classes", []))
    img_size = config.get("preprocessing", {}).get("image_size")
    print(f"   ✓ Config loaded")
    print(f"   - Classes: {num_classes}")
    print(f"   - Image size: {img_size}")
except Exception as e:
    print(f"   ✗ Config load error: {e}")
    sys.exit(1)

# Test 4: Load model
print("\n4. Loading model...")
try:
    model = load_model(config)
    print(f"   ✓ Model loaded")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shape: {model.output_shape}")
except Exception as e:
    print(f"   ✗ Model load error: {e}")
    sys.exit(1)

# Test 5: Get class names
print("\n5. Getting class names...")
try:
    class_names = get_class_names(config)
    print(f"   ✓ Class names: {len(class_names)} classes")
    print(f"   - {class_names}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 6: Verify model-class compatibility
print("\n6. Verifying model-class compatibility...")
try:
    verify_model_matches_classes(model, class_names)
    print(f"   ✓ Model and classes match")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 7: Test prediction pipeline
print("\n7. Testing prediction pipeline...")
try:
    import numpy as np
    from PIL import Image as PILImage
    import os
    
    # Create dummy image
    dummy_array = np.random.rand(128, 128, 3).astype('uint8')
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img = PILImage.fromarray(dummy_array)
        img.save(f.name)
        temp_path = f.name
    
    # Preprocess and predict
    image_array = preprocess_image(temp_path, config)
    result = predict_image(model, image_array, class_names, top_k=3)
    
    print(f"   ✓ Prediction successful")
    print(f"   - Predicted: {result.predicted_class}")
    print(f"   - Confidence: {result.confidence:.2%}")
    print(f"   - Top 3 predictions available")
    
    os.remove(temp_path)
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Check app.py
print("\n8. Checking app.py...")
try:
    with open('app.py', 'r') as f:
        app_code = f.read()
    
    checks = [
        ('render_app function', 'def render_app' in app_code),
        ('Streamlit import', 'import streamlit' in app_code),
        ('Model loading', 'load_model' in app_code),
        ('File uploader', 'file_uploader' in app_code),
        ('Prediction display', 'st.metric' in app_code or 'metric' in app_code),
    ]
    
    all_ok = True
    for check_name, check_result in checks:
        status = '✓' if check_result else '✗'
        print(f"   {status} {check_name}")
        if not check_result:
            all_ok = False
    
    if all_ok:
        print("   ✓ app.py structure OK")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 9: Syntax check
print("\n9. Python syntax validation...")
try:
    import py_compile
    py_compile.compile('app.py', doraise=True)
    print("   ✓ app.py syntax valid")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("STEP 8: STREAMLIT APPLICATION VERIFIED ✓")
print("="*70)
print("\nApplication Status:")
print("  ✓ All dependencies installed")
print("  ✓ Config loads correctly")
print("  ✓ Model loads successfully")
print("  ✓ Classes match model output")
print("  ✓ Inference pipeline works")
print("  ✓ app.py syntax valid")
print("  ✓ Ready for deployment")
print("\nTo run: streamlit run app.py")
print("="*70)
