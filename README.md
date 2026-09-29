# Skillset-Go-EduTech-AIML-Week2-Deep-Learning-with-PyTorch

A complete, executed submission for **Week 2 (Model-Building Phase)** of the Skillset Go EduTech AI/ML 4-Week Practical Learning Track — a feed-forward neural network, a CNN image classifier, controlled hyperparameter experiments, and a technical report, all built in PyTorch.

## Project Overview

Week 2 moves from classical ML (Week 1) into deep learning: building networks from tensors and autograd upward, training a CNN on image data, systematically comparing hyperparameter choices, and reporting the findings.

Three notebooks and one PDF report, each mapped to one official roadmap task, all executed end-to-end — every printed value, chart, and table is a real, reproducible result.

## Key Features

- Feed-forward network built from tensors, autograd, a custom `Dataset`, `DataLoader`, and a full train/validation loop
- Trained directly on `breast_cancer_cleaned.csv` — the same cleaned dataset from Week 1 — linking the two weeks into one pipeline
- CNN (two conv–ReLU–maxpool blocks) trained on image data, with training curves, a classification report, and a confusion matrix
- 15 controlled experiments (optimizer, learning rate, batch size, epochs), one factor varied at a time with fixed seeds for a fair comparison
- 2-page technical report generated programmatically from the experiment log
- Google Colab / Jupyter / VS Code compatible, CPU-only

## Technologies Used

- **Python** — Programming language
- **PyTorch** — Tensors, autograd, `nn.Module`, training loops
- **NumPy / Pandas** — Numerical operations and experiment logging
- **Matplotlib** — Visualization
- **Scikit-learn** — Splitting, preprocessing, metrics, bundled image dataset
- **ReportLab** — Automated PDF report generation

## Task Workflow

**1. Neural Network in PyTorch**
A 3-layer feed-forward network (ReLU, dropout) built with tensors and autograd. A custom `Dataset`/`DataLoader` feeds the Week 1 cleaned breast cancer data through a training loop (`BCEWithLogitsLoss`, Adam), tracked across epochs and evaluated on a held-out test set.

**2. CNN Image Classifier**
Image data is normalized and fed through a CNN — two conv/ReLU/max-pool blocks plus a dense classification head — trained with cross-entropy loss and Adam. Reported via training curves, a per-class classification report, and a confusion matrix; the trained model is saved.

**3. Hyperparameter Experimentation**
From a fixed baseline, one factor at a time — optimizer, learning rate, batch size, epochs — is varied across 15 runs on the same seed and split. Results are logged, ranked by validation accuracy, and visualized in a comparison chart.

**4. Deep Learning Technical Report**
A 2-page PDF — setup, model results, full comparison table, per-hyperparameter findings, and limitations — generated directly from `experiment_log.csv`, so it always reflects the actual runs.

## Project Structure

```
Task-2.1_Deep-Learning-with-PyTorch/
│
├── 1_neural_network_pytorch.ipynb
├── 2_cnn_image_classifier.ipynb
├── 3_hyperparameter_experiments.ipynb
├── deep_learning_technical_report.pdf
├── breast_cancer_cleaned.csv
├── cnn_digits.pt
├── experiment_log.csv
└── README.md
```

## Installation

```
git clone https://github.com/Prithyadarshan/Task-2.1_Deep-Learning-with-PyTorch.git
cd Task-2.1_Deep-Learning-with-PyTorch
pip install torch numpy pandas matplotlib scikit-learn jupyter reportlab
```

## How to Run

**Step 1:** Open any `.ipynb` in Google Colab, Jupyter, or VS Code.
**Step 2:** Run cells top to bottom. Keep `breast_cancer_cleaned.csv` in the same folder — Notebook 1 loads it directly.
**Step 3:** Each notebook prints its evaluation inline; Notebooks 2 and 3 also save charts and logs. The PDF is generated separately from the resulting experiment log.

## Output

- **`cnn_digits.pt`** — trained CNN weights from Task 2
- **`experiment_log.csv`** — all 15 hyperparameter runs: configuration, training loss, validation and test accuracy
- **`deep_learning_technical_report.pdf`** — 2-page summary of results, comparison table, and findings

## Applications

- Neural network training/evaluation workflows in PyTorch
- Systematic hyperparameter comparison methodology
- Baseline groundwork for CNN-based medical/histopathology image classification
- Foundation for the TensorFlow and GenAI tasks in Weeks 3–4

## Concepts Demonstrated

Tensors & Autograd · Custom Datasets/DataLoaders · Feed-Forward & Convolutional Architectures · Loss Functions & Optimizers · Training/Validation Loops · Hyperparameter Experimentation · Model Evaluation Metrics · Technical Reporting

## Future Enhancements

- Multi-seed runs to quantify variance
- Joint (grid) hyperparameter search instead of one-factor-at-a-time
- Transfer learning and augmentation on larger, real-world image data
- Cross-validation for the feed-forward network
- Extending the CNN toward the histopathology classification research project

## Limitations

- CNN trained on a small, bundled 8x8 digit dataset, not large-scale real-world images
- Each configuration run once with a single seed; 1–2 point differences may be noise
- Factors varied one at a time — interaction effects (e.g. learning rate × batch size) untested

## Learning Outcome

End-to-end experience building and training neural networks in PyTorch — from tensors and autograd through custom pipelines, convolutional architectures, and systematic hyperparameter evaluation, closing with a data-driven technical report. Extends directly from Week 1's classical ML foundations into the deep learning and applied AI work ahead.
