# Chest X-ray Pneumonia Classification (2-Class: NORMAL vs PNEUMONIA)

This repository contains a deep learning project for **binary classification** of chest X-ray images into:

- **NORMAL**
- **PNEUMONIA**

The system is built using **PyTorch**, includes a **custom Hybrid CNN with Residual Blocks**, and provides a fully interactive **Streamlit web application** with **Grad-CAM visualization** for model interpretability.

**Important:**  
This project is for **educational and research purposes only**.  
It is **not intended for clinical or medical diagnosis**.  
Always consult licensed medical professionals for medical interpretation of radiological images.


---

## Repository Structure


```bash
Pneumonia-HybridCNN/
│
├── model.py # Hybrid CNN + Residual Blocks + GradCAM
├── train.py # Full training script
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
├── environment.yml # Conda environment (optional)
├── README.md # Documentation
│
└── data/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/
```

---

## Features

### Hybrid CNN Architecture

The model combines:
- Deep convolutional layers  
- Residual skip connections  
- Adaptive pooling  
- A fully-connected classification head  

This structure enables strong feature extraction while avoiding vanishing gradients.

### Grad-CAM Explainability

The project includes a Grad-CAM implementation that generates:
- Heatmaps highlighting important regions
- Overlays on top of the original X-ray
- Layer selection for deeper inspection

### Streamlit Web Application

The included Streamlit interface provides:
- Image upload  
- Model prediction and class probabilities  
- Confidence indicators  
- Grad-CAM visualization  
- Medical disclaimers  
- Layer selection for interpretability  
- Side-by-side comparison views  

### Two-Class Disease Detection

The model classifies:
- Normal  
- Pneumonia  


Works with any dataset that follows the proper folder structure.

---

## Installation

### Clone the repository:

```bash
git clone https://github.com/yourusername/Pneumonia-HybridCNN.git
cd Pneumonia-HybridCNN
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

OR

```bash
conda env create -f environment.yml
conda activate pneumonia_cnn_env
```

### Data

The data can be pulled from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## Dataset Preparation

Your dataset must be strucutred as following:

```bash
data/train/NORMAL/
data/train/PNEUMONIA/

data/val/NORMAL/
data/val/PNEUMONIA/

data/test/NORMAL/
data/test/PNEUMONIA/

```

Supported formats: (.jpg, .jpeg, .png)
the training script automatically splits validation data from the training set.

---

## Training

Run:

```bash
python train.py
```

What happens during training:
- Data augmentation
- Class weighting for imbalance
- Learning rate scheduling
- Early stopping
- Best-model checkpointing
- Epoch-level performance reporting
- Test-set evaluation (confusion matrix + classification report)

---

## Running the Streamlit Application

Start the app:

```bash
streamlit rum app.py
```

Once launched, the interface allows you to:
- Upload a chest X-ray
- Generate predictions
- View per-class probabilities
- Inspect Grad-CAM heatmaps
- Switch Grad-CAM layers
- See educational disclaimers on every important step

---

## Model Performance

In typical training runs, the model achieves:
- About 96% overall accuracy
- Balanced performance across all three classes
- Strong recall for Pneumonia.
- Smooth generalization due to augmentation and class weighting

Performance depends on dataset composition and size.

---

## File Descriptions

### model.py

- HybridPneumoniaCNN model class
- ResidualBlock implementation
- Grad-CAM generation utilities

### train.py

- Train/validation/test loops
- Augmentation and dataset loaders
- Weighted loss
- Scheduler and early stopping
- Model saving and evaluation

### app.py

- Image loading and preprocessing
- Model inference
- Bar-graph probabilities
- Gradient-based heatmaps
- Color-coded confidence warnings
- Educational disclaimers

### requirements.txt

List of packages required for training and deployment.

---

## Contributing

Contributing

Contributions are welcome.

Possible improvements:
- Faster model variants
- Additional explainability methods
- Deployment scripts
- UI enhancements
- Support for more datasets

Submit a pull request or open an issue for discussion.

---

## Final Note

This repository is intended solely for educational experimentation with deep learning, medical imaging, model interpretability, and web deployment.

It is not a clinical diagnostic tool under any circumstances.