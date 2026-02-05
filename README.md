---
title: Drug Likeness Predictor
emoji: 💊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# 🧪 Drug-Likeness Predictor

> **A deep learning-based web application for predicting drug-likeness of chemical compounds using SMILES notation**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Performance Metrics](#-performance-metrics)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔬 Overview

This project was developed as part of the **BCA Data Analytics 30th Annual Project** (created on **April 4, 2025**) by **Arjun V** (Register Number: **221BCADA36**) to demonstrate the application of deep learning in pharmaceutical research. The Drug-Likeness Predictor uses a hybrid CNN-BiLSTM-LSTM neural network to predict whether a chemical compound exhibits drug-like properties based on its SMILES (Simplified Molecular Input Line Entry System) representation.

### What is Drug-Likeness?

Drug-likeness is a qualitative concept used in drug design to assess how "drug-like" a molecule is with respect to factors like bioavailability. This tool helps:
- **Pharmaceutical researchers** screen large compound libraries
- **Medicinal chemists** optimize lead compounds
- **Students and educators** understand structure-activity relationships
- **Drug discovery teams** prioritize synthesis candidates

---

## ✨ Features

### Core Functionality
- ✅ **SMILES Input Validation** - Accepts and validates molecular SMILES strings
- 🧠 **Deep Learning Prediction** - Hybrid CNN + Bidirectional LSTM + LSTM architecture
- 📊 **Confidence Scores** - Real-time prediction probability scores (0-1 scale)
- 🔬 **2D Visualization** - Interactive 2D molecular structure rendering using RDKit
- 🌐 **3D Visualization** - Rotatable 3D molecular viewer powered by Py3Dmol
- ⚡ **Fast Response** - Optimized inference pipeline (<100ms per prediction)
- 🎨 **Clean UI** - Modern, responsive web interface

### Technical Features
- **One-hot encoding** of SMILES sequences
- **Batch prediction** support
- **RESTful API** endpoints
- **Cross-platform** compatibility
- **Lightweight deployment** (1.2 MB model size)

---

## 🎬 Demo

![Drug-Likeness Predictor Demo](static/demo.gif)

### Example Predictions

**Drug-Like Molecules:**
```
CC(C)Cc1ccc(cc1)C(C)C(O)=O         # Ibuprofen - QED: 0.95
Cn1cnc(CNC(=O)N2CCC[C@H]2c2ccc(F)cc2)n1  # QED: 0.95
```

**Non-Drug-Like Molecules:**
```
O=C(CCCC[C@H]1SC[C@@H]2NC(=O)N[C@@H]21)N/N=C/c1ccccc1  # QED: 0.12
O=C(COc1c(F)c(F)c(F)c(F)c1F)Oc1ccc2c(c1)O/C(=C/C(=O)O)C(=O)c1ccccc12  # QED: 0.13
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB+ RAM recommended
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/Arj0010/drug-likeness-predictor.git
cd drug-likeness-predictor
```

### Step 2: Create Virtual Environment

**Using conda (Recommended):**
```bash
conda create -n drug_predictor python=3.10
conda activate drug_predictor
```

**Using venv:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** RDKit installation might take a few minutes. If you encounter issues, try:
```bash
conda install -c conda-forge rdkit  # For conda users
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; import rdkit; print('Installation successful!')"
```

---

## 💡 Usage

### Running the Application

1. **Start the Flask server:**
```bash
python app.py
```

2. **Open your browser and navigate to:**
```
http://127.0.0.1:5000
```

3. **Enter a SMILES string** in the input field and click "Predict"

### Using the Web Interface

1. **Input:** Enter a valid SMILES string (e.g., `CC(C)Cc1ccc(cc1)C(C)C(O)=O`)
2. **Predict:** Click the "Predict" button
3. **View Results:**
   - Prediction label (Drug-Like ✅ or Non-Drug-Like ⚠️)
   - Confidence score (0.00 to 1.00)
   - 2D molecular structure
4. **3D Visualization:** Click "View 3D Structure" for interactive 3D model

### Command Line Testing

Test the API using curl:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O"}'
```

---

## 🏗️ Model Architecture

### Neural Network Design

The model uses a **hybrid deep learning architecture** combining:

```python
Sequential([
    # Feature Extraction Layer
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(71, 89)),
    MaxPooling1D(pool_size=2),
    
    # Bidirectional Sequence Processing
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    
    # Temporal Pattern Recognition
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    
    # Classification Layers
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### Architecture Components

| Layer | Purpose | Details |
|-------|---------|---------|
| **Conv1D** | Local feature extraction | Captures molecular substructures |
| **Bidirectional LSTM** | Bidirectional dependencies | Processes sequences forward & backward |
| **LSTM** | Temporal patterns | Captures long-range molecular interactions |
| **Dropout (30%)** | Regularization | Prevents overfitting |
| **Dense (Sigmoid)** | Binary classification | Outputs drug-likeness probability |

### Input Processing

- **Input:** SMILES string (variable length)
- **Encoding:** One-hot encoding
- **Max Length:** 71 characters
- **Vocabulary Size:** 89 unique tokens
- **Output:** Drug-likeness probability (0-1)

---

## 📊 Dataset

### Source
- **Database:** [ZINC Database](https://zinc.docking.org/)
- **Size:** 250,000 random molecules
- **Split:** 80% training, 20% validation
- **Features:** SMILES, logP, QED, SAS

### Data Preprocessing

1. **Validation:** Remove invalid SMILES
2. **Canonicalization:** Standardize molecular representations
3. **Tokenization:** Break SMILES into atomic tokens
4. **Augmentation:** Generate randomized SMILES variants
5. **Encoding:** One-hot encode sequences

### QED Score

The model predicts **Quantitative Estimate of Drug-likeness (QED)**, which considers:
- Molecular weight
- LogP (lipophilicity)
- Number of hydrogen bond donors/acceptors
- Polar surface area
- Number of rotatable bonds
- Number of aromatic rings
- Structural alerts

---

## 📁 Project Structure

```
drug-likeness-predictor/
│
├── app.py                      # Flask application entry point
├── best_model.keras            # Pre-trained TensorFlow model (1.2 MB)
├── tokenizer.pkl               # SMILES tokenizer (vocabulary mapping)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
│
├── templates/
│   ├── index.html             # Main prediction interface
│   └── visualize.html         # 3D visualization page
│
├── static/
│   └── screenshots/           # Application screenshots
│
├── docs/
│   ├── ARCHITECTURE.md        # System architecture documentation
│   └── PROJECT_SHOWCASE.md    # Challenge → Solution → Impact
│
├── Model_Creation_1.ipynb     # Model training notebook
├── Assess_help.txt            # Sample SMILES for testing
│
└── DATA_README.md             # Dataset information
```

---

## 🔌 API Documentation

### Endpoints

#### POST `/predict`

Predicts drug-likeness for a given SMILES string.

**Request:**
```json
{
  "smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
}
```

**Response:**
```json
{
  "smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
  "prediction": "Drug-Like ✅",
  "score": 0.95,
  "image_2d": "base64_encoded_image_string"
}
```

**Error Response:**
```json
{
  "error": "Invalid SMILES string!"
}
```

#### GET `/visualize_3d`

Displays 3D molecular structure.

**Parameters:**
- `smiles` (string): SMILES representation

**Returns:** HTML page with interactive 3D viewer

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | ~85% |
| **ROC-AUC Score** | ~0.89 |
| **Dataset Size** | 250,000 molecules |
| **Training Time** | ~2 hours (GPU) |
| **Inference Speed** | <100ms per prediction |
| **Model Size** | 1.2 MB |
| **Vocabulary Size** | 89 tokens |
| **Max Sequence Length** | 71 characters |

### Key Observations

- **logP and SAS** are more discriminative features than QED alone
- **Outliers in logP** had minimal impact after normalization
- **Hybrid architecture** (CNN + BiLSTM + LSTM) effectively captures structural patterns

### Performance Characteristics

- ✅ Balanced performance on drug-like and non-drug-like compounds
- ✅ Robust to various molecular structures
- ✅ Handles complex SMILES notation including stereochemistry
- ✅ Efficient memory usage
- ✅ Fast inference suitable for real-time applications

---

## 🔧 Troubleshooting

### Common Issues

#### 1. RDKit Installation Fails

**Problem:** `pip install rdkit-pypi` fails or takes too long

**Solution:**
```bash
# Use conda instead
conda install -c conda-forge rdkit
```

#### 2. TensorFlow GPU Issues

**Problem:** TensorFlow not detecting GPU

**Solution:**
```bash
# Check CUDA installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CUDA-compatible TensorFlow
pip install tensorflow[and-cuda]
```

#### 3. Port Already in Use

**Problem:** `Address already in use: 127.0.0.1:5000`

**Solution:**
```bash
# Change port in app.py
app.run(debug=True, port=5001)

# Or kill existing process (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

#### 4. Invalid SMILES Error

**Problem:** Getting "Invalid SMILES string!" error

**Solution:**
- Verify SMILES syntax using online validators
- Check for special characters
- Try canonical SMILES from PubChem or ChemSpider
- Use examples from `Assess_help.txt`

#### 5. Model Loading Error

**Problem:** `Unable to load model` error

**Solution:**
```bash
# Verify model file exists
ls -la best_model.keras

# Check TensorFlow version compatibility
pip install tensorflow==2.15.0
```

---

## 🚀 Future Improvements

### Planned Enhancements

#### 1. **Advanced ML Features**
- [ ] Implement attention mechanisms for better interpretability
- [ ] Add ensemble methods (Random Forest, XGBoost) for comparison
- [ ] Implement transfer learning from pre-trained chemical models
- [ ] Add uncertainty quantification using Bayesian neural networks

#### 2. **Extended Predictions**
- [ ] Multi-task learning for simultaneous prediction of:
  - Solubility
  - Toxicity
  - Blood-brain barrier permeability
  - Metabolic stability
- [ ] ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) predictions
- [ ] Synthetic accessibility scoring

#### 3. **User Interface Enhancements**
- [ ] Batch upload functionality (CSV/SDF files)
- [ ] Export results to PDF/Excel
- [ ] Comparison mode for multiple compounds
- [ ] Interactive molecular editor
- [ ] Dark mode theme
- [ ] Mobile-responsive design improvements

#### 4. **Performance Optimization**
- [ ] Implement caching for frequent queries
- [ ] Add GPU acceleration for batch predictions
- [ ] Optimize model for edge deployment (TensorFlow Lite)
- [ ] Implement asynchronous processing for large batches

#### 5. **Data & Model Improvements**
- [ ] Expand training dataset to 1M+ molecules
- [ ] Include more diverse chemical scaffolds
- [ ] Implement active learning for continuous improvement
- [ ] Add explainable AI (SHAP, LIME) for prediction interpretation
- [ ] Cross-validation with external datasets (ChEMBL, PubChem)

#### 6. **Deployment & Infrastructure**
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] RESTful API with authentication
- [ ] Rate limiting and usage analytics
- [ ] Continuous integration/deployment pipeline

#### 7. **Scientific Features**
- [ ] Molecular descriptor calculation
- [ ] Similarity search against known drugs
- [ ] Substructure highlighting for drug-like features
- [ ] Generate optimization suggestions
- [ ] Integration with molecular docking tools

#### 8. **Documentation & Testing**
- [ ] Comprehensive unit tests (pytest)
- [ ] Integration tests for API endpoints
- [ ] Performance benchmarking suite
- [ ] User documentation and tutorials
- [ ] API documentation (Swagger/OpenAPI)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Getting Started

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🧪 Additional test cases
- 🎨 UI/UX enhancements

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Datasets & Tools
- **[ZINC Database](https://zinc.docking.org/)** - Molecular dataset source
- **[RDKit](https://www.rdkit.org/)** - Open-source cheminformatics toolkit
- **[TensorFlow](https://www.tensorflow.org/)** - Deep learning framework
- **[Py3Dmol](https://3dmol.csb.pitt.edu/)** - 3D molecular visualization library

### References
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [Quantitative Estimation of Drug-likeness (QED)](https://www.nature.com/articles/nchem.1243)
- [SMILES Tutorial](http://opensmiles.org/opensmiles.html)
- [Deep Learning for Drug Discovery](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00266)

### Academic Context
This project was completed as part of the **BCA Data Analytics 30th Annual Project** at [Your College Name], demonstrating the practical application of deep learning techniques in pharmaceutical research and drug discovery.

---

## 👨‍💻 Author

**Arjun V (Arjun Vavullipathy)**

- **Register Number:** 221BCADA36
- **GitHub:** [@Arj0010](https://github.com/Arj0010)
- **LinkedIn:** [Arjun Vavullipathy](https://www.linkedin.com/in/arjun-vavullipathy-722877196/)
- Email: arjunvavullipathy@gmail.com
---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. Predictions should not be used as the sole basis for drug development decisions. Always consult with qualified medicinal chemists and conduct proper laboratory testing before making any pharmaceutical decisions.

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/Arj0010/drug-likeness-predictor/issues)
3. Open a new issue with detailed information
4. Contact the author via LinkedIn

---

**Made with ❤️ for the scientific community**

*Advancing drug discovery through artificial intelligence*
