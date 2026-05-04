
---

# GSN: Graph-Semantic-Net
### Real-Time Vocal Extraction via Harmonic-Aware Refinement

**GSN** is a hybrid vocal separation framework that combines **Physical Intelligence** (Harmonic Graphs) and **Linguistic Intelligence** (CLAP Semantic Steering). By injecting an acoustic physics prior into a frozen Demucs backbone, GSN achieves high-fidelity refinement with a Real-Time Factor (RTF) of **0.27**.



---

##  Key Features
* **Harmonic Graph Refinement:** A custom GCN whose adjacency matrix $A$ is hard-wired with integer-ratio overtone relationships ($2f_0, 3f_0, \dots$).
* **Zero-Shot Semantic Steering:** Uses frozen **CLAP** text embeddings to define extraction targets via natural language (e.g., *"clean lead vocals"*).
* **Real-Time Performance:** Engineered for low-latency applications (11.6ms algorithmic latency) on commodity hardware.
* **Asymmetric Compute:** High-weight encoders (CLAP) run once per prompt; only the lightweight GSN refiner runs per frame.

---

##  Results (MUSDB18-HQ)
GSN provides a measurable boost in separation quality over standard baselines while remaining significantly faster than Transformer-based models.

| Model | SI-SDR (dB) | SIR (dB) | RTF (GPU) | Latency |
| :--- | :---: | :---: | :---: | :---: |
| U-Net Baseline | 3.22 | 5.10 | **0.05** | 11.6ms |
| Demucs v3 (Base) | 5.12 | 7.85 | 0.22 | 11.6ms |
| **Hybrid GSN (Ours)** | **5.25** | **8.15** | 0.27 | **11.6ms** |
| HT-Demucs | 8.90 | 12.4 | 1.85 | >150ms |

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GSN-Vocal-Extraction.git
cd GSN-Vocal-Extraction

# Install dependencies
pip install -r requirements.txt
```

---

##  Usage

### 1. Separation with Semantic Prompting
You can guide the separation using natural language. The CLAP embeddings will steer the GCN to focus on the described target.

```python
from gsn import GSNInference

model = GSNInference(checkpoint="weights/gsn_best.pt")

# Perform separation with a semantic prompt
model.separate(
    input_path="mixture.wav",
    output_path="vocals_refined.wav",
    prompt="high-pitched female lead vocals, dry studio recording"
)
```

### 2. Real-Time Stream
The GSN refiner is designed for block-based processing.
```bash
python stream_gsn.py --input_device 1 --prompt "lead vocals"
```

---

##  Methodology
The core innovation is the **Harmonic Adjacency Matrix**. Unlike standard CNNs that look at neighboring pixels, our GCN propagates energy along the harmonic ladder.



1.  **Stage 1:** Coarse structural separation via **Demucs**.
2.  **Stage 2:** Semantic vector generation via **CLAP**.
3.  **Stage 3:** Residual refinement via **Harmonic GCN** ($2.8\text{M}$ parameters).

---

## 📜 Citation
If you use this work in your research, please cite:
```bibtex
@article{saini2026gsn,
  title={GSN: Graph-Semantic-Net for Real-Time Vocal Extraction via Harmonic-Aware Refinement},
  author={Saini, Disha},
  year={2026},
  journal={NIELIT Technical Research}
}
```
