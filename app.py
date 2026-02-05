import logging
from typing import Optional, Tuple

import numpy as np
import joblib
import gradio as gr
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = "best_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

# Load resources
try:
    logger.info("Loading model and tokenizer...")
    model = tf.keras.models.load_model(MODEL_PATH)
    token_to_index = joblib.load(TOKENIZER_PATH)
    logger.info("Resources loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load resources: {e}")
    raise RuntimeError("Critical resources missing. Check files.")

vocab_size = len(token_to_index)
max_length = 71


# ---------------------------------------------------------------------------
# Core logic (reused from Flask version)
# ---------------------------------------------------------------------------

def one_hot_encode_smiles(smiles: str) -> np.ndarray:
    """One-hot encode a single SMILES string."""
    tokens = list(smiles)
    one_hot_vector = np.zeros((max_length, vocab_size), dtype=np.float32)
    for j, token in enumerate(tokens[:max_length]):
        if token in token_to_index:
            one_hot_vector[j, token_to_index[token]] = 1
    return one_hot_vector.reshape(1, max_length, vocab_size)


def visualize_2D(smiles: str):
    """Return a PIL Image of the 2D molecular structure, or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Draw.MolToImage(mol, size=(400, 400))
    except Exception as e:
        logger.error(f"Error generating 2D image: {e}")
        return None


def convert_smiles_to_mol(smiles: str) -> Optional[str]:
    """Convert SMILES to a MOL block for 3D visualisation."""
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


def build_3d_html(mol_block: str) -> str:
    """Return an HTML snippet with an embedded 3Dmol.js viewer."""
    escaped = mol_block.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    return f"""
    <div id="viewer-container" style="width:100%;display:flex;justify-content:center;">
        <div id="mol-viewer" style="width:500px;height:400px;border:1px solid #ccc;border-radius:8px;"></div>
    </div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        (function() {{
            var viewer = $3Dmol.createViewer(document.getElementById("mol-viewer"), {{
                backgroundColor: "white"
            }});
            var molData = `{escaped}`;
            viewer.addModel(molData, "mol");
            viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{scale: 0.25}}}});
            viewer.zoomTo();
            viewer.render();
            viewer.spin("y", 1);
        }})();
    </script>
    """


# ---------------------------------------------------------------------------
# Prediction wrapper
# ---------------------------------------------------------------------------

def predict(smiles: str) -> Tuple[str, object, str]:
    """
    Run prediction and return (label_text, pil_image_or_None, html_3d_or_empty).
    """
    smiles = smiles.strip()
    if not smiles:
        return "Please enter a SMILES string.", None, ""

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return "Invalid SMILES string! Please check your input.", None, ""

    encoded = one_hot_encode_smiles(smiles)
    try:
        score = float(model.predict(encoded, verbose=0).flatten()[0])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return "Prediction failed. Please try again.", None, ""

    label = "Drug-Like" if score >= 0.5 else "Non-Drug-Like"
    result_text = f"{label}  (confidence: {score:.2f})"

    img = visualize_2D(smiles)

    mol_block = convert_smiles_to_mol(smiles)
    html_3d = build_3d_html(mol_block) if mol_block else ""

    return result_text, img, html_3d


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

example_smiles = [
    ["CC(=O)OC1=CC=CC=C1C(=O)O"],        # Aspirin
    ["CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"],  # Testosterone
    ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],    # Caffeine
    ["C1=CC=CC=C1"],                       # Benzene (non-drug-like)
]

with gr.Blocks(theme=gr.themes.Soft(), title="Drug-Likeness Predictor") as demo:
    gr.Markdown(
        """
        # Drug-Likeness Predictor
        Predict whether a molecule is **drug-like** using a deep-learning model
        (CNN + BiLSTM + LSTM) trained on 250 000 molecules from the ZINC database.

        Enter a **SMILES** string below and click **Predict Drug-Likeness**.
        """
    )

    with gr.Row():
        with gr.Column():
            smiles_input = gr.Textbox(
                label="SMILES Input",
                placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)",
                lines=1,
            )
            predict_btn = gr.Button("Predict Drug-Likeness", variant="primary")
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction", interactive=False)
            image_output = gr.Image(label="2D Molecular Structure", type="pil")

    viewer_html = gr.HTML(label="3D Interactive Viewer")

    gr.Examples(
        examples=example_smiles,
        inputs=[smiles_input],
        label="Example Molecules",
    )

    predict_btn.click(
        fn=predict,
        inputs=[smiles_input],
        outputs=[prediction_output, image_output, viewer_html],
    )
    smiles_input.submit(
        fn=predict,
        inputs=[smiles_input],
        outputs=[prediction_output, image_output, viewer_html],
    )

if __name__ == "__main__":
    demo.launch()
