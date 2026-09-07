# 🧠 EEG Analysis

A Python-based EEG signal analysis pipeline for extracting spectral and nonlinear features from preprocessed EEG data using **MNE-Python**, **NumPy**, **Pandas**, and **AntroPy**.

This repository is designed as a modular EEG analysis pipeline that will be progressively extended with additional signal-processing, connectivity, statistical, and machine-learning analyses.

---

## 🔬 Current Analyses

### 1. Power Spectral Density (PSD)

Power Spectral Density is estimated using **Welch's method** from 1–30 Hz.

- Welch PSD estimation
- 2-second EEG epochs
- `n_fft = 256`
- 50% overlap (`n_overlap = 128`)
- Frequency resolution ≈ 0.98 Hz

### 2. Band Power

Four canonical EEG frequency bands are analyzed:

| Band | Frequency Range |
|---|---:|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–12 Hz |
| Beta | 12–30 Hz |

For each EEG channel, the pipeline calculates:

- **Absolute Band Power**
- **Relative Band Power**
- Mean power across epochs
- Standard deviation across epochs

Relative power is calculated as:

**Relative Power = Band Power / Total Power (1–30 Hz)**

### 3. EEG Visualization

The pipeline generates:

- PSD plots across EEG channels
- Log-scaled PSD visualization
- Frequency-band topographic maps

### 4. Hjorth Parameters

Three Hjorth parameters are extracted from each EEG channel:

- **Activity** — variance of the EEG signal
- **Mobility** — measure related to the frequency characteristics of the signal
- **Complexity** — measure of the similarity of the signal to a pure sine wave

Features are calculated per epoch and then summarized across epochs.

### 5. Sample Entropy

**Sample Entropy (SampEn)** is calculated to quantify the regularity and complexity of EEG signals.

Parameters:

- Order (`m`) = 2
- Chebyshev distance metric

Sample entropy is calculated for each epoch and summarized per EEG channel.

---

## 📊 Feature Output

The pipeline produces structured feature tables containing:

- Subject ID
- Session
- EEG channel
- Frequency band
- Absolute power
- Relative power
- Hjorth Activity
- Hjorth Mobility
- Hjorth Complexity
- Sample Entropy

These features can subsequently be used for:

- Statistical analysis
- Pre/Post intervention comparisons
- Correlation with behavioral measures
- Machine learning
- Computational neuroscience analyses

---

## 🛠️ Technologies

- Python
- MNE-Python
- NumPy
- Pandas
- Matplotlib
- AntroPy

---

## 📁 Current Structure

```text
EEG-Analysis/
│
├── spectral_analysis/
│   └── eeg_feature_extraction.ipynb
│
└── README.md
```

The repository will be expanded as additional EEG analyses are implemented.
