# SceneGuard: Training-Time Voice Protection with Scene-Consistent Audible Background Noise

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **SceneGuard**, a novel voice protection method that defends against unauthorized voice cloning by mixing scene-consistent audible background noise during training time.

---

## 🎯 Overview

SceneGuard protects voice recordings from unauthorized voice cloning attacks by:
- **Scene-Consistent Noise**: Using audible background noise that matches the acoustic scene (e.g., cafe sounds for cafe recordings)
- **Gradient-Based Optimization**: Jointly optimizing temporal mask and noise strength to minimize speaker similarity
- **Robustness**: Resistant to audio preprocessing like denoising, compression, and filtering
- **Usability**: Maintains high speech quality with STOI > 0.98 and WER < 4%

Unlike imperceptible perturbation-based methods, SceneGuard leverages the naturalness of scene-consistent background noise to achieve both strong protection and robustness to countermeasures.

---

## 📊 Key Results

| Metric | Clean Training | SceneGuard | Improvement |
|--------|----------------|------------|-------------|
| Speaker Similarity ↓ | 1.000 | **0.945** | **5.5% reduction** |
| Word Error Rate (%) | 0.0 | 3.6 | +3.6% |
| STOI Score | 1.00 | 0.986 | -0.014 |
| Robustness to Denoising | - | **Enhanced** | Similarity drops to 0.745 |

**Statistical Significance**: p < 10⁻¹⁵, Cohen's d = 2.18 (very large effect)

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f ENVIRONMENT.yml
conda activate SceneGuard

# Or using pip
pip install torch torchaudio speechbrain whisper pesq pystoi seaborn
```

### 2. Data Preparation

**Speech Data** (e.g., LibriTTS):
```bash
# Download LibriTTS
wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz -C data/raw/speech/
```

**Noise Library** (TAU Urban Acoustic Scenes):
```bash
# Download TAU dataset
# Visit: https://zenodo.org/record/6337421
# Extract to: data/raw/noise_lib/

# Build scene-organized noise library
python scripts/build_noise_library.py
```

### 3. Model Downloads

**Speaker Verification** (ECAPA-TDNN):
```python
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/sv/ecapa-tdnn"
)
```

**ASR** (Whisper):
```python
import whisper
model = whisper.load_model("base", download_root="models/asr/")
```

### 4. Scene Classification

```bash
# Label speech files with scene classifications
python scripts/label_scenes.py \
    --input_dir data/raw/speech/libritts/ \
    --output_file data/interim/scene_labels.csv
```

### 5. Defense Generation

**Direct Mixing** (baseline):
```bash
python scripts/run_defense_mixing.py \
    --mode direct \
    --input_dir data/raw/speech/ \
    --output_dir data/processed/defended/
```

**Optimized Mixing** (recommended):
```bash
python scripts/run_defense_mixing.py \
    --mode optimized \
    --max-epochs 50 \
    --lambda-sim 1.0 \
    --lambda-reg 0.01 \
    --input_dir data/raw/speech/ \
    --output_dir data/processed/defended_optimized/
```

### 6. Evaluation

```bash
# Evaluate defense effectiveness
python scripts/evaluate_defense.py \
    --clean_dir data/raw/speech/ \
    --defended_dir data/processed/defended_optimized/ \
    --output_file reports/metrics/defense_evaluation.csv
```

---

## 📁 Project Structure

```
SceneGuard/
├── src/sceneguard/          # Core implementation
│   ├── mixing/              # Noise mixing module
│   │   └── mixer.py         # SceneGuardMixer (direct + optimized)
│   ├── optimization/        # Gradient-based optimization
│   │   ├── optimizer.py     # SceneGuardOptimizer
│   │   └── losses.py        # Loss functions (SIM, REG)
│   ├── utils/               # Utilities
│   │   └── vad.py           # Voice Activity Detection
│   └── eval/                # Evaluation metrics
│       └── metrics.py       # SIM, WER, PESQ, STOI, MCD
│
├── scripts/                 # Experiment scripts
│   ├── build_noise_library.py
│   ├── label_scenes.py
│   ├── run_defense_mixing.py
│   ├── evaluate_defense.py
│   ├── test_optimizer.py
│   ├── visualize_optimization.py
│   └── figures/             # Figure generation scripts
│
├── tests/                   # Unit tests
│   ├── test_ecapa.py
│   └── test_whisper.py
│
├── paper/                   # LaTeX paper
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── references.bib
│
├── data/                    # Data directories (empty, user-provided)
├── models/                  # Model checkpoints (empty, user-downloaded)
├── results/                 # Experiment results
├── reports/                 # Evaluation reports
└── artifacts/               # Model cards and logs

```

---

## 🔬 Method Overview

SceneGuard protects voice recordings through three key steps:

### 1. Scene-Consistent Noise Selection
- Classify speech recordings into acoustic scenes (e.g., "cafe", "street", "office")
- Sample background noise from the same scene category
- Ensures noise is natural and contextually appropriate

### 2. Gradient-Based Optimization
Jointly optimize temporal mask `m(t)` and noise strength `γ`:

```
x'(t) = x(t) + γ · m(t) · n(t)
```

**Objective Function**:
```
min  λ_SIM · cos(e(x'), e(x)) + λ_REG · R(m, γ)
m,γ

subject to: SNR ∈ [10, 20] dB
```

Where:
- `e(·)`: ECAPA-TDNN speaker encoder
- `cos(·,·)`: Cosine similarity
- `R(·)`: Regularization (mask smoothness + γ magnitude)

### 3. Automatic Constraint Satisfaction
- Bounded reparameterization ensures SNR stays in [10, 20] dB
- No manual tuning required
- Converges in ~15 seconds per sample (RTX A6000)

**Key Results**:
- Final Speaker Similarity: **-0.378 ± 0.110** (negative = strong protection)
- SNR: **18.51 ± 0.04 dB** (extremely stable, σ = 0.04 dB)
- SNR Variance Reduction: **98.6%** (from 2.9 to 0.04 dB)

---

## 📈 Performance

### Training Attack Protection
| Method | Speaker SIM ↓ | WER (%) | STOI | PESQ |
|--------|---------------|---------|------|------|
| Clean | 1.000 | 0.0 | 1.00 | 4.64 |
| Random Noise | 0.965 | 5.8 | 0.97 | 1.85 |
| Gaussian Noise | 0.968 | 5.2 | 0.98 | 1.92 |
| **SceneGuard** | **0.945** | **2.77** | **0.99** | 2.22 |

### Robustness to Countermeasures
| Countermeasure | SIM | Change | Protection |
|----------------|-----|--------|------------|
| None (baseline) | 0.937 | - | ✓ |
| MP3 128 kbps | 0.901 | -0.036 | ✓ Maintained |
| Spectral Subtraction | **0.745** | **-0.192** | ✓ **Enhanced** |
| Lowpass 3400 Hz | **0.704** | **-0.232** | ✓ **Enhanced** |
| Downsample 8 kHz | **0.688** | **-0.249** | ✓ **Enhanced** |

**Finding**: Aggressive preprocessing (denoising, filtering) **enhances** protection by damaging speech more than noise!

### Zero-Shot Attack
| Reference | Mean SIM | Success Rate (SIM > 0.7) |
|-----------|----------|--------------------------|
| Clean | 0.618 | 20.0% |
| **Defended** | **0.588** | **13.3%** (**-33.5%**) |

---

## 🎨 Visualization

### Optimization Trajectory
```bash
# Visualize optimization process
python scripts/visualize_optimization.py \
    --sample-id <sample_name> \
    --output-dir paper/figures/optimization_viz/
```

Generates:
- Loss curves (Total, Speaker Similarity, Regularization)
- SNR trajectory (with constraint bounds)
- Waveform comparison (Original → Protected)
- Spectrogram comparison

---

## 🧪 Experiments

### Baseline Comparison
```bash
python scripts/baseline_comparison.py \
    --input_dir data/raw/speech/ \
    --output_dir results/baselines/
```

### SNR Ablation Study
```bash
python scripts/snr_ablation.py \
    --snr_ranges "5,10" "10,20" "15,25" "20,30" \
    --output_dir results/ablation/
```

### Optimization Analysis
```bash
python scripts/analyze_optimization_comparison.py
```

---

## 📦 Dependencies

**Core**:
- `torch >= 2.0.0`
- `torchaudio >= 2.0.0`
- `speechbrain >= 0.5.16` (ECAPA-TDNN)
- `openai-whisper >= 20230314` (ASR)

**Metrics**:
- `pesq >= 0.0.4`
- `pystoi >= 0.3.3`

**Visualization**:
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`

**Utilities**:
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `pandas >= 2.0.0`
- `librosa >= 0.10.0`
- `soundfile >= 0.12.0`

See `ENVIRONMENT.yml` for complete environment specification.

---

## 📄 Citation

If you use SceneGuard in your research, please cite:

```bibtex
@article{sceneguard2025,
  title={SceneGuard: Training-Time Voice Protection with Scene-Consistent Audible Background Noise},
  author={[Authors]},
  journal={[Conference/Journal]},
  year={2025}
}
```

---

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or collaborations:
- **Email**: [Your Email]
- **GitHub Issues**: https://github.com/richael-sang/SceneGuard/issues

---

## 🙏 Acknowledgments

- [LibriTTS](https://www.openslr.org/60/) for speech data
- [TAU Urban Acoustic Scenes](https://zenodo.org/record/6337421) for scene noise
- [SpeechBrain](https://speechbrain.github.io/) for ECAPA-TDNN
- [OpenAI Whisper](https://github.com/openai/whisper) for ASR
- [SafeSpeech](https://github.com/wxzyd123/SafeSpeech) for training infrastructure inspiration

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Work

- **SafeSpeech**: Imperceptible perturbation-based voice protection
- **AntiFake**: Audio adversarial examples for deepfake detection
- **VoiceGuard**: Privacy-preserving voice authentication
- **Audio Adversarial Examples**: Defenses against audio machine learning

**Key Difference**: SceneGuard uses **audible, scene-consistent noise** instead of imperceptible perturbations, achieving superior robustness to audio preprocessing while maintaining natural sound quality.

---

**⭐ Star us on GitHub if you find SceneGuard useful!**

