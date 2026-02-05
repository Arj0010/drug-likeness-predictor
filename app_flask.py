import logging
import base64
from io import BytesIO
from typing import Optional, Dict, Any, Union, Tuple

import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_caching import Cache
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import tensorflow as tf


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Application configuration."""
    MODEL_PATH = "best_model.keras"
    TOKENIZER_PATH = "tokenizer.pkl"
    CACHE_TYPE = "SimpleCache"  # Simple in-memory cache
    CACHE_DEFAULT_TIMEOUT = 300 # 5 minutes

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize Cache
cache = Cache(app)

# Load resources
try:
    logger.info("Loading model and tokenizer...")
    model = tf.keras.models.load_model(Config.MODEL_PATH)
    token_to_index = joblib.load(Config.TOKENIZER_PATH)
    logger.info("Resources loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load resources: {e}")
    raise RuntimeError("Critical resources missing. check files.")

vocab_size = len(token_to_index)
max_length = 71

def one_hot_encode_smiles(smiles: str) -> np.ndarray:
    """
    One-hot encode a single SMILES string.
    
    Args:
        smiles: The SMILES string to encode.
        
    Returns:
        A numpy array of shape (1, max_length, vocab_size).
    """
    tokens = list(smiles)
    one_hot_vector = np.zeros((max_length, vocab_size), dtype=np.float32)

    for j, token in enumerate(tokens[:max_length]):
        if token in token_to_index:
            token_idx = token_to_index[token]
            one_hot_vector[j, token_idx] = 1

    return one_hot_vector.reshape(1, max_length, vocab_size)

def visualize_2D(smiles: str) -> Optional[str]:
    """
    Generate 2D molecular structure and return as base64-encoded string.
    
    Args:
        smiles: SMILES string.
        
    Returns:
        Base64 string of the PNG image or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            img = Draw.MolToImage(mol, size=(300, 300))
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error generating 2D image: {e}")
            return None
    return None

def convert_smiles_to_mol(smiles: str) -> Optional[str]:
    """
    Convert SMILES to MOL block for 3D visualization.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        return Chem.MolToMolBlock(mol)
    except Exception as e:
        logger.error(f"Error converting to 3D MOL block: {e}")
        return None

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict drug-likeness from user input SMILES string."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input format"}), 400
            
        smiles = data.get("smiles", "").strip()
        if not smiles:
            return jsonify({"error": "SMILES string is required"}), 400

        # Check for cached prediction (manually or via function separation if needed)
        # For simple POST with JSON, @cache.cached needs configuration or manual key generation.
        # Here we will cache the computationally expensive part: predict_logic
        
        result = get_prediction(smiles)
        
        if "error" in result:
             return jsonify(result), 400
             
        return jsonify(result)

    except Exception as e:
        logger.exception("Unexpected error in /predict")
        return jsonify({"error": "Internal server error"}), 500

@cache.memoize(timeout=300)
def get_prediction(smiles: str) -> Dict[str, Any]:
    """
    Core prediction logic, cached based on SMILES input.
    """
    logger.info(f"Processing prediction for SMILES: {smiles}")
    
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        logger.warning(f"Invalid SMILES received: {smiles}")
        return {"error": "Invalid SMILES string!"}

    # Preprocess
    encoded_input = one_hot_encode_smiles(smiles)

    # Inference
    try:
        prediction = model.predict(encoded_input, verbose=0).flatten()[0]
        label = "Drug-Like ✅" if prediction >= 0.5 else "Non-Drug-Like ⚠️"
        
        # Generates image
        img_2d = visualize_2D(smiles)
        
        return {
            "smiles": smiles,
            "prediction": label,
            "score": round(float(prediction), 2),
            "image_2d": img_2d
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": "Prediction calculation failed"}

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