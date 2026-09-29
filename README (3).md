# Skillset-Go-EduTech-AIML-Week2-Deep-Learning-PyTorch

Week 2 (Model-Building Phase) submission for the Skillset Go EduTech AI/ML 4-Week Practical Learning Track: a feed-forward network, a CNN image classifier, controlled hyperparameter experiments, and a short technical report, all in PyTorch.

## Project Overview

Week 2 moves from classical ML (Week 1) to deep learning. All notebooks are executed end-to-end, so every number, chart, and table is a real result. Random seeds and splits are fixed, so runs are reproducible on CPU.

## Task Summary

| Task | File | Result |
|---|---|---|
| 2.1 Neural Network in PyTorch | `1_neural_network_pytorch.ipynb` | Feed-forward net on `breast_cancer_cleaned.csv` (Week 1 output): 95.35% test accuracy, F1 0.962 |
| 2.2 CNN Image Classifier | `2_cnn_image_classifier.ipynb` | Two-block CNN on handwritten-digit images: 97.41% test accuracy; model saved as `cnn_digits.pt` |
| 2.3 Hyperparameter Experimentation | `3_hyperparameter_experiments.ipynb` | 15 controlled runs (optimizer, learning rate, batch size, epochs); log in `experiment_log.csv`; best: Adam, lr 1e-3, batch 64, 40 epochs (97.4% val, 98.2% test) |
| 2.4 Technical Report | `deep_learning_technical_report.pdf` | 2-page report comparing experiments |

## Technologies Used

- **Python**, **PyTorch** (tensors, autograd, `Dataset`/`DataLoader`, `nn.Module`)
- **NumPy**, **Pandas**, **Matplotlib**, **Scikit-learn** (splits, metrics, bundled digits dataset)
- **ReportLab** (PDF report), **Jupyter Notebook / Google Colab**

## Project Structure

```
Week2-Deep-Learning-PyTorch/
├── 1_neural_network_pytorch.ipynb
├── 2_cnn_image_classifier.ipynb
├── 3_hyperparameter_experiments.ipynb
├── deep_learning_technical_report.pdf
├── make_report.py
├── breast_cancer_cleaned.csv
├── experiment_log.csv
├── experiment_comparison_table.csv
├── ffnn_breast_cancer.pt
├── cnn_digits.pt
└── *.png  (training curves, confusion matrix, sample images, comparison chart)
```

## How to Run

```
pip install torch numpy pandas matplotlib scikit-learn jupyter reportlab
```

Run the notebooks in order (`1` to `3`), then `python make_report.py` to regenerate the PDF from `experiment_log.csv`. Keep `breast_cancer_cleaned.csv` (from Week 1) in the same folder as notebook 1.

## Key Findings

- Learning rate and training length mattered most; batch size mattered little.
- SGD at Adam's learning rate (1e-3) did not learn; with larger rates it trained but stayed below Adam within 20 epochs.
- Each configuration ran once with one seed, and validation/test sets have 270 images each, so differences of 1-2 points are within noise.

## Limitations

- The CNN uses the small 8x8 scikit-learn digits set, so results do not transfer directly to real-world (for example histopathology) images.
- One factor was varied at a time, so interactions between hyperparameters were not tested.

## Future Enhancements

- Repeat runs over several seeds and report mean and spread
- Joint learning-rate x batch-size grid
- Larger image datasets with augmentation and transfer learning
