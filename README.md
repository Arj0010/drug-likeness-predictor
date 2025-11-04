# 🧪 Drug-Likeness Prediction Using Deep Learning

A **deep learning-based web application** that predicts the **drug-likeness** of chemical compounds from their **SMILES (Simplified Molecular Input Line Entry System)** representation. Features real-time prediction with interactive **2D and 3D molecular visualization**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Drug-Likeness Predictor](static/demo.gif)

---

## 🔍 Features

- ✅ **SMILES Input Validation** - Accepts and validates molecular SMILES strings
- 🧠 **Deep Learning Prediction** - CNN + LSTM + Bidirectional RNN architecture
- 📊 **Confidence Scores** - Real-time prediction probability scores
- 🔬 **2D Visualization** - Interactive 2D molecular structure rendering (RDKit)
- 🌐 **3D Visualization** - Rotatable 3D molecular viewer (Py3Dmol)
- ⚡ **Fast Response** - Optimized inference pipeline
- 🎨 **Clean UI** - Modern, responsive web interface

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Arj0010/drug-likeness-predictor.git
   cd drug-likeness-predictor
   ```

2. **Create and activate virtual environment** (recommended)
   ```bash
   # Using conda
   conda create -n drug_predictor python=3.10
   conda activate drug_predictor

   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 💡 Usage

### Example SMILES Inputs

**Drug-Like Molecules:**
```
CC(C)Cc1ccc(cc1)C(C)C(O)=O         # Ibuprofen
Cn1cnc(CNC(=O)N2CCC[C@H]2c2ccc(F)cc2)n1
CN(C[C@H]1CCCCO1)C(=O)c1cc2ccccc2c(=O)[nH]1
```

**Non-Drug-Like Molecules:**
```
CC(=O)Oc1ccccc1C(=O)O              # Aspirin (in this model's context)
O=C(CCCC[C@H]1SC[C@@H]2NC(=O)N[C@@H]21)N/N=C/c1ccccc1
O=C(COc1c(F)c(F)c(F)c(F)c1F)Oc1ccc2c(c1)O/C(=C/C(=O)O)C(=O)c1ccccc12
```

### Using the Web Interface

1. Enter a valid SMILES string in the input field
2. Click "Predict" button
3. View the prediction result with confidence score
4. Examine the 2D molecular structure
5. Click "View 3D Structure" for interactive 3D visualization

---

## 🏗️ Model Architecture

The model uses a hybrid deep learning architecture combining:

```python
Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(71, 89)),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

**Key Components:**
- **Conv1D Layer**: Extracts local features from SMILES sequences
- **Bidirectional LSTM**: Captures long-range dependencies in both directions
- **LSTM Layer**: Further processes temporal patterns
- **Dropout Layers**: Prevents overfitting (30% rate)
- **Dense Layers**: Final classification with sigmoid activation

**Training Details:**
- Dataset: 250,000 molecules from ZINC database
- Training Accuracy: ~87%
- Input: One-hot encoded SMILES (max length: 71, vocab size: 89)
- Output: Drug-likeness probability (0-1)

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Flask 3.0+ | Web server & API endpoints |
| **ML Framework** | TensorFlow 2.15 | Deep learning model |
| **Chemistry** | RDKit | Molecular parsing & 2D visualization |
| **3D Viewer** | Py3Dmol | Interactive 3D molecular structures |
| **Frontend** | HTML/CSS/JavaScript | User interface |
| **Data Processing** | NumPy, Pandas | Data manipulation |
| **Serialization** | Joblib | Model & tokenizer storage |

---

## 📁 Project Structure

```
drug-likeness-predictor/
│
├── app.py                      # Flask application entry point
├── best_model.keras            # Pre-trained TensorFlow model
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
├── Model_Creation_1.ipynb     # Model training notebook
└── Assess_help.txt            # Sample SMILES for testing
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 87% |
| **Dataset Size** | 250,000 molecules |
| **Training Time** | ~2 hours (GPU) |
| **Inference Speed** | <100ms per prediction |
| **Model Size** | 1.2 MB |

**Performance Characteristics:**
- Balanced performance on drug-like and non-drug-like compounds
- Robust to various molecular structures
- Handles complex SMILES notation including stereochemistry

---

## 🔧 Development

### Running Tests

Test with sample molecules:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O"}'
```

### API Endpoints

**POST /predict**
- Input: `{"smiles": "SMILES_STRING"}`
- Output: JSON with prediction, score, and 2D image

**GET /visualize_3d**
- Parameter: `smiles=SMILES_STRING`
- Output: HTML page with 3D molecular viewer

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ZINC Database** - Molecular dataset source
- **RDKit** - Open-source cheminformatics toolkit
- **TensorFlow** - Deep learning framework
- **Py3Dmol** - 3D molecular visualization library

---

## 📚 References

- [RDKit Documentation](https://www.rdkit.org/docs/)
- [ZINC Database](https://zinc.docking.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Py3Dmol Documentation](https://3dmol.csb.pitt.edu/)
- [Quantitative Estimation of Drug-likeness (QED)](https://www.nature.com/articles/nchem.1243)

---

## 👨‍💻 Author

**Arjun Vavullipathy**

- GitHub: [@Arj0010](https://github.com/Arj0010)
- LinkedIn: [Arjun Vavullipathy](https://www.linkedin.com/in/arjun-vavullipathy-722877196/)

---

## ⚠️ Disclaimer

This tool is for research and educational purposes only. Predictions should not be used as the sole basis for drug development decisions. Always consult with qualified chemists and conduct proper laboratory testing.

---

**Made with ❤️ for the scientific community**
