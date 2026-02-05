/**
 * Drug-Likeness Predictor - Core Logic
 */

class DrugLikenessApp {
    constructor() {
        this.smilesInput = document.getElementById('smilesInput');
        this.predictBtn = document.getElementById('predictBtn');
        this.resultContainer = document.getElementById('resultContainer');
        this.loadingOverlay = document.getElementById('loadingOverlay');

        this.initEventListeners();
    }

    initEventListeners() {
        this.predictBtn.addEventListener('click', () => this.handlePrediction());

        // Allow Enter key to submit
        this.smilesInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handlePrediction();
        });
    }

    setLoading(isLoading) {
        if (isLoading) {
            this.loadingOverlay.classList.add('active');
            this.predictBtn.disabled = true;
        } else {
            this.loadingOverlay.classList.remove('active');
            this.predictBtn.disabled = false;
        }
    }

    async handlePrediction() {
        const smiles = this.smilesInput.value.trim();

        if (!smiles) {
            this.showError("Please enter a valid SMILES string.");
            return;
        }

        this.setLoading(true);
        this.clearResults();

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ smiles: smiles }),
            });

            const result = await response.json();

            if (response.ok) {
                this.displayResults(result);
            } else {
                this.showError(result.error || "An error occurred during prediction.");
            }
        } catch (error) {
            console.error(error);
            this.showError("Network error. Please try again.");
        } finally {
            this.setLoading(false);
        }
    }

    displayResults(data) {
        const isDrugLike = data.prediction.includes("Drug-Like");

        const html = `
            <div class="result-card">
                <div class="metric-row">
                    <span class="label">Prediction</span>
                    <span class="value ${isDrugLike ? 'drug-like' : 'non-drug-like'}">
                        ${data.prediction}
                    </span>
                </div>
                
                <div class="metric-row">
                    <span class="label">Confidence Score</span>
                    <span class="value">${data.score}</span>
                </div>

                <div class="metric-row">
                     <span class="label">SMILES</span>
                     <span class="value" style="font-family: monospace; font-size: 0.9rem;">
                        ${data.smiles.length > 30 ? data.smiles.substring(0, 30) + '...' : data.smiles}
                     </span>
                </div>
                
                <div class="molecule-view">
                    <div class="label" style="margin-bottom: 0.5rem;">Molecular Structure (2D)</div>
                    <img src="data:image/png;base64,${data.image_2d}" alt="2D Structure" class="molecule-img">
                    
                    <a href="/visualize_3d?smiles=${encodeURIComponent(data.smiles)}" target="_blank" class="view-3d-btn">
                        <span>View 3D Model</span>
                        <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                    </a>
                </div>
            </div>
        `;

        this.resultContainer.innerHTML = html;
        this.resultContainer.classList.add('visible');
    }

    showError(message) {
        this.resultContainer.innerHTML = `
            <div class="result-card" style="border-color: var(--danger);">
                <div class="metric-row" style="border-bottom: none; margin-bottom: 0;">
                    <span class="label" style="color: var(--danger);">Error</span>
                    <span class="value" style="color: var(--danger); font-size: 1rem;">${message}</span>
                </div>
            </div>
        `;
        this.resultContainer.classList.add('visible');
    }

    clearResults() {
        this.resultContainer.classList.remove('visible');
        setTimeout(() => {
            this.resultContainer.innerHTML = '';
        }, 300);
    }
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    new DrugLikenessApp();
});
