import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

# Initialize Flask app
app = Flask(__name__)

# Load trained model and tokenizer
model_path = "best_model.keras"
model = load_model(model_path)

tokenizer_path = "tokenizer.pkl"
token_to_index = joblib.load(tokenizer_path)

# Load other necessary files
vocab_size = len(token_to_index)
max_length = 71  # Ensure this matches the training configuration


def one_hot_encode_smiles(smiles):
    """One-hot encode a single SMILES string."""
    tokens = list(smiles)
    one_hot_vector = np.zeros((max_length, vocab_size), dtype=np.float32)

    for j, token in enumerate(tokens[:max_length]):  # Limit sequence length
        if token in token_to_index:
            token_idx = token_to_index[token]
            one_hot_vector[j, token_idx] = 1

    return one_hot_vector.reshape(1, max_length, vocab_size)


def visualize_2D(smiles):
    """Generate 2D molecular structure and return as base64-encoded string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    return None


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict drug-likeness from user input SMILES string."""
    data = request.get_json()
    smiles = data.get("smiles", "").strip()

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string!"}), 400

    # Preprocess SMILES
    encoded_input = one_hot_encode_smiles(smiles)

    # Run inference
    prediction = model.predict(encoded_input).flatten()[0]
    label = "Drug-Like ✅" if prediction >= 0.5 else "Non-Drug-Like ⚠️"

    # Generate 2D structure
    img_2d = visualize_2D(smiles)

    return jsonify({
        "smiles": smiles,
        "prediction": label,
        "score": round(float(prediction), 2),
        "image_2d": img_2d
    })


def convert_smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES
    mol = Chem.AddHs(mol)  # Add hydrogens
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Generate 3D coordinates
    return Chem.MolToMolBlock(mol)  # Convert to MOL format


@app.route('/visualize_3d')
def visualize_3d():
    smiles = request.args.get('smiles', '').strip()

    if not smiles:
        return "❌ No SMILES input provided!", 400

    mol_block = convert_smiles_to_mol(smiles)
    if mol_block is None:
        return "❌ Invalid SMILES input!", 400

    return render_template("visualize.html", mol_block=mol_block)


if __name__ == "__main__":
    app.run(debug=True)
s