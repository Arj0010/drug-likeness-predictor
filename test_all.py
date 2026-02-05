"""
Comprehensive test suite for the Drug-Likeness Predictor.
Tests model loading, encoding, validation, prediction, and visualization.
"""
import os
import sys
import traceback
import base64
from io import BytesIO

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("DRUG-LIKENESS PREDICTOR - UNIT TESTS")
print("=" * 60)
print()

passed = 0
failed = 0
total = 0


def test(name, func):
    global passed, failed, total
    total += 1
    try:
        result = func()
        if result:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} - returned False")
            failed += 1
    except Exception as e:
        print(f"  FAIL: {name} - {e}")
        traceback.print_exc()
        failed += 1


# ========================================
# TEST 1: Model Loading
# ========================================
print("[1] Model Loading Tests")
print("-" * 40)

import tensorflow as tf
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

model = None
token_to_index = None


def test_model_load():
    global model
    model = tf.keras.models.load_model("best_model.keras")
    return model is not None


def test_tokenizer_load():
    global token_to_index
    token_to_index = joblib.load("tokenizer.pkl")
    return token_to_index is not None and len(token_to_index) > 0


def test_tokenizer_is_dict():
    return isinstance(token_to_index, dict)


def test_model_input_shape():
    input_shape = model.input_shape
    return input_shape[1] == 71


def test_model_output_sigmoid():
    # Output should be (None, 1) for binary classification
    output_shape = model.output_shape
    return output_shape[-1] == 1


test("Model loads from best_model.keras", test_model_load)
test("Tokenizer loads from tokenizer.pkl", test_tokenizer_load)
test("Tokenizer is a dictionary", test_tokenizer_is_dict)
test("Model input shape has max_length=71", test_model_input_shape)
test("Model output shape is single neuron (binary)", test_model_output_sigmoid)

print()
print(f"  Model input shape: {model.input_shape}")
print(f"  Model output shape: {model.output_shape}")
print(f"  Tokenizer vocab size: {len(token_to_index)}")

# ========================================
# TEST 2: Encoding Tests
# ========================================
print()
print("[2] SMILES Encoding Tests")
print("-" * 40)

vocab_size = len(token_to_index)
max_length = 71


def one_hot_encode_smiles(smiles):
    tokens = list(smiles)
    one_hot_vector = np.zeros((max_length, vocab_size), dtype=np.float32)
    for j, token in enumerate(tokens[:max_length]):
        if token in token_to_index:
            one_hot_vector[j, token_to_index[token]] = 1
    return one_hot_vector.reshape(1, max_length, vocab_size)


def test_encoding_shape():
    encoded = one_hot_encode_smiles("CCO")
    return encoded.shape == (1, 71, vocab_size)


def test_encoding_nonzero():
    encoded = one_hot_encode_smiles("CCO")
    return np.sum(encoded) > 0


def test_empty_smiles_encoding():
    encoded = one_hot_encode_smiles("")
    return encoded.shape == (1, 71, vocab_size) and np.sum(encoded) == 0


def test_long_smiles_truncation():
    long_smiles = "C" * 100
    encoded = one_hot_encode_smiles(long_smiles)
    return encoded.shape == (1, 71, vocab_size)


def test_one_hot_correctness():
    # Each position should have at most one 1
    encoded = one_hot_encode_smiles("CCO")
    for i in range(max_length):
        if np.sum(encoded[0, i, :]) > 1:
            return False
    return True


test("Encoding produces correct shape", test_encoding_shape)
test("Encoding has non-zero values for valid SMILES", test_encoding_nonzero)
test("Empty SMILES encodes to zeros", test_empty_smiles_encoding)
test("Long SMILES truncated to max_length", test_long_smiles_truncation)
test("One-hot encoding is correct (max one 1 per position)", test_one_hot_correctness)

# ========================================
# TEST 3: SMILES Validation with RDKit
# ========================================
print()
print("[3] SMILES Validation Tests (RDKit)")
print("-" * 40)


def test_valid_smiles_aspirin():
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    return mol is not None


def test_valid_smiles_caffeine():
    mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    return mol is not None


def test_valid_smiles_ibuprofen():
    mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(O)=O")
    return mol is not None


def test_invalid_smiles():
    mol = Chem.MolFromSmiles("XYZ123!!!")
    return mol is None


def test_benzene():
    mol = Chem.MolFromSmiles("C1=CC=CC=C1")
    return mol is not None


test("Aspirin SMILES is valid", test_valid_smiles_aspirin)
test("Caffeine SMILES is valid", test_valid_smiles_caffeine)
test("Ibuprofen SMILES is valid", test_valid_smiles_ibuprofen)
test("Invalid SMILES returns None", test_invalid_smiles)
test("Benzene SMILES is valid", test_benzene)

# ========================================
# TEST 4: Prediction Tests
# ========================================
print()
print("[4] Prediction Tests")
print("-" * 40)


def make_prediction(smiles):
    encoded = one_hot_encode_smiles(smiles)
    pred = model.predict(encoded, verbose=0).flatten()[0]
    return float(pred)


# Known test molecules
drug_like_smiles = [
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(O)=O"),
    ("Drug-like 1", "Cn1cnc(CNC(=O)N2CCC[C@H]2c2ccc(F)cc2)n1"),
    ("Drug-like 2", "O=C(Cn1ccc(C(F)(F)F)n1)NN1Cc2ccccc2C1"),
]

non_drug_like_smiles = [
    ("Benzene", "C1=CC=CC=C1"),
    ("Non-drug 1", "O=C(CCCC[C@H]1SC[C@@H]2NC(=O)N[C@@H]21)N/N=C/c1ccccc1"),
]

print()
print("  Drug-like molecules:")
for name, smi in drug_like_smiles:
    score = make_prediction(smi)
    label = "Drug-Like" if score >= 0.5 else "Non-Drug-Like"
    print(f"    {name}: score={score:.4f} -> {label}")

print()
print("  Non-drug-like molecules:")
for name, smi in non_drug_like_smiles:
    score = make_prediction(smi)
    label = "Drug-Like" if score >= 0.5 else "Non-Drug-Like"
    print(f"    {name}: score={score:.4f} -> {label}")


def test_prediction_range():
    score = make_prediction("CC(=O)OC1=CC=CC=C1C(=O)O")
    return 0.0 <= score <= 1.0


def test_prediction_deterministic():
    s1 = make_prediction("CC(=O)OC1=CC=CC=C1C(=O)O")
    s2 = make_prediction("CC(=O)OC1=CC=CC=C1C(=O)O")
    return abs(s1 - s2) < 1e-5


def test_different_molecules_different_scores():
    s1 = make_prediction("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
    s2 = make_prediction("C1=CC=CC=C1")  # Benzene
    return s1 != s2


test("Prediction score in [0, 1] range", test_prediction_range)
test("Predictions are deterministic", test_prediction_deterministic)
test("Different molecules get different scores", test_different_molecules_different_scores)

# ========================================
# TEST 5: 2D Visualization Tests
# ========================================
print()
print("[5] 2D Visualization Tests")
print("-" * 40)


def visualize_2D(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    return None


def test_2d_valid_smiles():
    img = visualize_2D("CC(=O)OC1=CC=CC=C1C(=O)O")
    return img is not None and len(img) > 100


def test_2d_invalid_smiles():
    img = visualize_2D("invalid_smiles_xyz!!!")
    return img is None


def test_2d_base64_valid():
    img = visualize_2D("CC(=O)OC1=CC=CC=C1C(=O)O")
    try:
        decoded = base64.b64decode(img)
        # PNG magic bytes
        return decoded[:4] == b"\x89PNG"
    except Exception:
        return False


test("2D image generated for valid SMILES", test_2d_valid_smiles)
test("2D image returns None for invalid SMILES", test_2d_invalid_smiles)
test("2D image is valid base64 PNG", test_2d_base64_valid)

# ========================================
# TEST 6: 3D MOL Block Conversion
# ========================================
print()
print("[6] 3D MOL Block Conversion Tests")
print("-" * 40)


def convert_smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return Chem.MolToMolBlock(mol)


def test_3d_valid():
    mol_block = convert_smiles_to_mol("CC(=O)OC1=CC=CC=C1C(=O)O")
    return mol_block is not None and "V2000" in mol_block


def test_3d_invalid():
    mol_block = convert_smiles_to_mol("invalid!!!")
    return mol_block is None


def test_3d_has_coordinates():
    mol_block = convert_smiles_to_mol("CCO")
    if mol_block is None:
        return False
    lines = mol_block.split("\n")
    # Check that atom lines have non-zero coordinates
    for line in lines[4:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                if x != 0 or y != 0 or z != 0:
                    return True
            except ValueError:
                continue
    return False


test("3D MOL block generated for valid SMILES", test_3d_valid)
test("3D MOL block returns None for invalid SMILES", test_3d_invalid)
test("3D MOL block contains 3D coordinates", test_3d_has_coordinates)

# ========================================
# SUMMARY
# ========================================
print()
print("=" * 60)
print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
print("=" * 60)
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"{failed} test(s) FAILED")
    sys.exit(1)
