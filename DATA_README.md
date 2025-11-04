# Dataset Information

## Missing Files

The following large dataset files are not included in this repository due to size constraints:

- `250k_rndm_zinc_drugs_clean_3.csv` (23 MB)
- `cleaned_dataset.csv` (96 MB)

## Dataset Source

The training data consists of 250,000 random molecules from the [ZINC Database](https://zinc.docking.org/).

## Obtaining the Dataset

If you need the original training data:

1. Visit the ZINC Database: https://zinc.docking.org/
2. Download a subset of drug-like molecules
3. Clean and preprocess the SMILES strings
4. Use the provided `Model_Creation_1.ipynb` notebook for reference on data preprocessing

## Note

The pre-trained model (`best_model.keras`) and tokenizer (`tokenizer.pkl`) are included, so you can use the application immediately without downloading the training data.

## Data Format

The CSV files contain the following columns:
- `smiles`: SMILES representation of the molecule
- `qed`: Quantitative Estimate of Drug-likeness score (0-1)
- Additional descriptors as needed

## Example SMILES

See `Assess_help.txt` for sample SMILES strings you can use for testing.
