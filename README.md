ChestXNet-PruneQuant — Project Report

Deep learning–based pneumonia detection from chest X-rays with pruning & quantization


Executive Summary

This project develops and evaluates an efficient deep-learning pipeline to classify chest X-ray images as Normal or Pneumonia. A pretrained ResNet50 backbone with a compact classification head was fine-tuned on a small dataset (378 images). To enable deployment on resource-constrained devices, the trained model was compressed using pruning and dynamic quantization. Key findings from an example run: baseline test accuracy ≈ 94.64%, pruned variants ≈ 96.43%, and quantized models preserved baseline accuracy while reducing model size for CPU inference.

1. Introduction

Early, accurate detection of pneumonia from chest radiographs can significantly improve patient outcomes. While deep neural networks deliver strong performance, real-world deployment often requires smaller, faster models. This work aims to train a high-quality classifier and demonstrate model compression approaches (pruning and quantization) that preserve—or sometimes improve—performance while making the model more practical for edge devices.

2. Dataset & Preprocessing
2.1 Dataset

Total images: 378

Normal: 200

Pneumonia: 178

Structure: standard ImageFolder layout (one folder per class).
![Class Distribution](figure/images/Class_Distribution.png)  
*Figure 1 — Class Distribution of Our Dataset.*

2.2 Splitting & Reproducibility

Train / Val / Test: 70% / 15% / 15% (seeded sampling for reproducibility).

Splits preserve class proportions.
![Split Data](figure/images/Dataset_split.png)  
*Figure 2 — Dataset Split into Train, Validation, Test.*

2.3 Preprocessing & Augmentation

Input size: 224 × 224 (ResNet50).

Training augmentations: random horizontal flip, small rotation (±10°), color jitter.

Normalization: ImageNet mean/std.

Imbalance handled via WeightedRandomSampler and class-weighted loss.

3. Model Architecture

Backbone: ResNet50 (pretrained on ImageNet).

Head: Linear(2048 → 512) → BatchNorm → ReLU → Dropout(0.5) → Linear(512 → 2).

Rationale: pretrained ResNet50 provides robust feature extraction; the lightweight head balances capacity and regularization for small datasets.

4. Training & Evaluation Protocol

Optimizer: Adam (lr = 1e-4)

Loss: CrossEntropy with class weights (or optional FocalLoss)

Batch size: 32; Epochs: 12 (example)

Metrics logged each epoch: accuracy, precision (macro), recall (macro), F1 (macro), AUC (macro), AUC-PR (macro), loss for both train and validation.

Best model selected by minimum validation loss and saved as best_model.pth.

5. Results
5.1 Training Behavior

Training converges quickly with augmentations and regularization. Validation metrics are stable with minor fluctuations expected due to the small validation set.

![Training Curves](figure/images/Training&validation_loss&accuracy_curves.png)  
*Figure 3 — Training vs Validation (Accuracy and Loss Curves).*

5.2 Test Performance (example run)

Baseline test accuracy: ≈ 94.64%

Example classification report (test set) demonstrates balanced precision and recall across classes; very few misclassifications occurred with only one critical false negative in the example run.

![Testing Accuracy](figure/images/testing_accuracy.png)  
*Figure 4 — Model's Classification Reports.*

![Confusion Matrix](figure/images/confusion_matrix.png)  
*Figure 5 — Confusion Matrix.*

6. Model Optimization
6.1 Pruning — what & why

Pruning removes weights or entire structures to reduce model size and computation. Two styles:

Unstructured pruning: removes individual low-magnitude weights (e.g., global L1). Produces sparse weight matrices; storage benefits require sparse support.

Structured pruning: removes whole filters/channels, directly reducing FLOPs and enabling faster dense computation.

Benefits: lower model size, possible regularization effect, potential inference speedups (with correct tooling).
Risks: aggressive pruning may remove informative weights and harm accuracy.

6.2 Pruning applied here

One-shot global L1 pruning (50%) applied across Conv and Linear weights.

Iterative pruning: repeated smaller pruning steps (3 × 20%) optionally with re-training between steps.

Per-layer sensitivity analysis identifies layers that tolerate pruning vs. those that are critical.

Reported example results:

One-shot (50%): ≈ 96.43% test accuracy.

Iterative (3×20%): ≈ 96.43% test accuracy.
Observation: pruning can act as a regularizer and sometimes improve accuracy on small datasets — verify across multiple seeds.

![Per Layer Sensitivity](figure/images/per_layer_pruning_sensitivity_analysis.png)  
*Figure 6 — Per Layer Pruning Sensitivity Analysis.*
![Baseline vs Pruning](figure/images/baselineVSpruning_accuracy_comparision.png)  
*Figure 7 — Baseline vs Pruning Accuracy comparision.*

6.3 Quantization — what & why

Quantization reduces numeric precision (float32 → int8), shrinking model size and often improving CPU inference speed. Approaches:

Dynamic quantization: quick post-training quantization ideal for Linear / LSTM layers.

Static quantization / QAT: better for convolutional networks but requires calibration or training.

Applied here: dynamic quantization of Linear layers (qint8), resulting in preserved accuracy (≈ baseline) with reduced size and faster CPU inference.


7. Comparative Summary (example run)
Model variant	Test accuracy
Baseline	≈ 94.64%
One-shot prune (50%)	≈ 96.43%
Iterative prune (3×20%)	≈ 96.43%
Dynamic quantized (int8)	≈ 94.64%

![Final comparison](figure/images/final_model_accuracy_comparision.png)  
*Figure 5 — Comparative Analysis.*

Interpretation: pruning and quantization provide complementary benefits — pruning can reduce overfitting and parameter redundancy, while quantization enables compact, fast CPU inference. Results are dataset-sensitive; repeat and average across seeds for robust claims.

9. Conclusions & Recommendations

A ResNet50 fine-tuned with careful augmentation and class-balance techniques yields high accuracy on the example dataset.

Pruning can produce compact models with preserved or improved accuracy when applied carefully (use per-layer sensitivity to guide pruning).

Dynamic quantization is an easy post-training step to reduce model size and improve CPU inference without additional training.

For production: perform multiple runs, cross-validation, measure latency on the target device (ARM CPU / mobile), and consider QAT / structured pruning to maximize runtime gains.
