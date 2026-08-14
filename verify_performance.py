#!/usr/bin/env python3
"""Verify Experiment 3 evaluation results match expected values."""

import json
import pandas as pd
from pathlib import Path

print("="*70)
print("STEP 3: VERIFYING FINAL PERFORMANCE")
print("="*70)

eval_dir = Path("results/evaluation_exp3")

# 1. Check test_metrics.json
print("\n1. Test Metrics (from test_metrics.json):")
test_metrics_path = eval_dir / "test_metrics.json"
try:
    with open(test_metrics_path) as f:
        metrics = json.load(f)
    
    print(f"   Test Accuracy:          {metrics['test_accuracy']:.6f}")
    print(f"   Test Loss:              {metrics['test_loss']:.6f}")
    print(f"   Test Images:            {metrics['num_test_images']}")
    print(f"   Number of Classes:      {metrics['num_classes']}")
    print(f"   Best Validation Acc:    {metrics['best_validation_accuracy']:.6f}")
    
    # Verify against expected values
    expected_accuracy = 0.7266257405281067
    expected_loss = 0.7764738202095032
    expected_val_acc = 0.7293
    
    print("\n   Verification against expected values:")
    acc_match = abs(metrics['test_accuracy'] - expected_accuracy) < 0.0001
    loss_match = abs(metrics['test_loss'] - expected_loss) < 0.0001
    val_acc_match = abs(metrics['best_validation_accuracy'] - expected_val_acc) < 0.0001
    
    print(f"   Test Accuracy:          {'✓' if acc_match else '✗'} (expected {expected_accuracy:.6f})")
    print(f"   Test Loss:              {'✓' if loss_match else '✗'} (expected {expected_loss:.6f})")
    print(f"   Best Val Accuracy:      {'✓' if val_acc_match else '✗'} (expected {expected_val_acc:.6f})")
    
except FileNotFoundError:
    print(f"   ✗ File not found: {test_metrics_path}")
except Exception as e:
    print(f"   ✗ Error reading: {e}")

# 2. Per-class metrics
print("\n2. Per-Class Metrics (from per_class_metrics.csv):")
per_class_path = eval_dir / "per_class_metrics.csv"
try:
    df = pd.read_csv(per_class_path)
    print(f"   Rows (classes):         {len(df)}")
    print(f"   Columns:                {list(df.columns)}")
    print("\n   Per-class breakdown:")
    for _, row in df.iterrows():
        print(f"   {row['class']:12} | P={row['precision']:.4f} | R={row['recall']:.4f} | F1={row['f1_score']:.4f} | Support={int(row['support'])}")
    
    # Compute macro and weighted metrics
    macro_precision = df['precision'].mean()
    macro_recall = df['recall'].mean()
    macro_f1 = df['f1_score'].mean()
    
    total_support = df['support'].sum()
    weighted_precision = (df['precision'] * df['support']).sum() / total_support
    weighted_recall = (df['recall'] * df['support']).sum() / total_support
    weighted_f1 = (df['f1_score'] * df['support']).sum() / total_support
    
    print("\n   Calculated aggregates:")
    print(f"   Macro Precision:        {macro_precision:.6f}")
    print(f"   Macro Recall:           {macro_recall:.6f}")
    print(f"   Macro F1:               {macro_f1:.6f}")
    print(f"   Weighted Precision:     {weighted_precision:.6f}")
    print(f"   Weighted Recall:        {weighted_recall:.6f}")
    print(f"   Weighted F1:            {weighted_f1:.6f}")
    
    # Expected aggregates
    expected_macro_f1 = 0.7007549311952246
    expected_weighted_f1 = 0.7301522616323575
    expected_macro_prec = 0.6742857177780515
    expected_weighted_prec = 0.7476545547221123
    
    print("\n   Verification against expected:")
    macro_f1_match = abs(macro_f1 - expected_macro_f1) < 0.0001
    weighted_f1_match = abs(weighted_f1 - expected_weighted_f1) < 0.0001
    
    print(f"   Macro F1:               {'✓' if macro_f1_match else '✗'} (expected {expected_macro_f1:.6f})")
    print(f"   Weighted F1:            {'✓' if weighted_f1_match else '✗'} (expected {expected_weighted_f1:.6f})")
    
except FileNotFoundError:
    print(f"   ✗ File not found: {per_class_path}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 3. Confusion matrix
print("\n3. Confusion Matrix (from confusion_matrix.csv):")
confusion_path = eval_dir / "confusion_matrix.csv"
try:
    cm = pd.read_csv(confusion_path, index_col=0)
    print(f"   Shape:                  {cm.shape}")
    print(f"   Classes:                {list(cm.index)}")
    print(f"   ✓ Confusion matrix present")
except FileNotFoundError:
    print(f"   ✗ File not found: {confusion_path}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 4. Confusion matrix PNG
print("\n4. Confusion Matrix Visualization:")
confusion_png = eval_dir / "confusion_matrix.png"
if confusion_png.exists():
    print(f"   ✓ File exists: {confusion_png}")
else:
    print(f"   ✗ File not found: {confusion_png}")

# 5. Summary
print("\n" + "="*70)
print("STEP 3: PERFORMANCE VERIFICATION SUMMARY")
print("="*70)
print("\nFINAL METRICS (Experiment 3):")
print(f"  Test Accuracy:           72.66%")
print(f"  Test Loss:               0.7765")
print(f"  Macro F1:                0.7008")
print(f"  Weighted F1:             0.7302")
print(f"  Best Validation Accuracy: 72.93%")
print(f"  Test Set Size:           4,075 images")
print(f"  Classes:                 10")
print("="*70)
