# GENETRACE // Forensic Paternity & STR Analytics Core

<div align="center">

**A forensic-grade paternity testing platform integrating classical STR analysis with machine learning-powered inference**

[![Python](https://img.shields.io/badge/Python-3.15-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Status](https://img.shields.io/badge/Status-Beta-orange.svg)]()

---

## ⚡ Quick Start

```bash
# Navigate to project directory
cd Biology

# Install dependencies
pip install -r requirements.txt

# Initialize ML server (runs on port 8000)
python ml_server.py

# Launch the applications
open index.html   # Interactive 3D STR Science & Educational Guide
open model.html   # GENETRACE Forensic Analytics & Clinical Reporting Workspace
```

---

## 🧬 Overview

GENETRACE is a dual-engine forensic paternity testing system designed for clinical and legal laboratories. It combines:

1. **Local Forensic Core** — A validated STR analysis engine implementing Mendelian inheritance rules, population genetics, and likelihood ratio calculations per CODIS standards
2. **ML Pipeline** — Random Forest classifier that predicts paternity outcomes from 180 genetic features (20 CODIS loci × 2 alleles + compatibility metrics)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GENETRACE System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐       ┌─────────────────────────┐     │
│  │   Frontend       │       │    ML Backend (API)      │     │
│  │  (model.html)    │◄─────►│   (ml_server.py)        │     │
│  │                  │       │                         │     │
│  │• Case Management │       │• Random Forest Model    │     │
│  │• STR Input Grid  │       │• Decision Tree Baseline │     │
│  │• Results Dashboard│      │• ONNX Inference         │     │
│  └────────┬─────────┘       └──────────┬──────────────┘     │
│           │                            │                     │
│           │ HTTP POST                  │                     │
│           ▼                            ▼                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Forensic STR Analysis Engine             │   │
│  ├───────────────────────────────────────────────────────┤   │
│  │• CODIS 20 Loci Compatibility Check                    │   │
│  │• Mendelian Inheritance Validation                     │   │
│  │• Mutation Detection (±1, ±0.2 alleles)                │   │
│  │• Population Frequency Lookup (Caucasian baseline)     │   │
│  │• Paternity Index (PI) per Locus                       │   │
│  │• Combined Paternity Index (CPI/LR)                    │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **20 CODIS Loci Analysis** | CSF1PO, D3S1358, D5S818, D7S820, D8S1179, D13S317, D16S539, D18S51, D21S11, FGA, TH01, TPOX, vWA, D1S1656, D2S441, D2S1338, D10S1248, D12S391, D19S433, D22S1045 |
| **Triad Analysis** | Mother-Child-Alleged Father genotype evaluation with obligate allele determination |
| **Mutation Detection** | Single-step STR mutations (±1 or ±0.2 repeat units) with reduced PI calculation |
| **Population Genetics** | Allele frequency databases for Caucasian, African American, Hispanic, and Asian populations |
| **Likelihood Ratio Engine** | Per-locus PI × Combined CPI with forensic interpretation thresholds |
| **ML Inference Layer** | Random Forest classifier (ROC-AUC validation + precision/recall metrics) |

### Analysis Metrics

- **Locus Compatibility**: Mendelian inheritance check (1 = compatible, 0 = incompatible)
- **Paternity Index (PI)**: Probability ratio per locus
- **Combined PI (CPI)**: Product of all locus PIs
- **Posterior Probability**: Bayesian calculation with default prior = 0.5
- **Verdict Classification**: STRONG INCLUSION / EXCLUSION / REVIEW REQUIRED / INCONCLUSIVE

---

## 📁 Project Structure

```
Biology/
├── index.html              # Interactive 3D STR Science & Educational Introduction (Three.js + GSAP)
├── model.html              # Complete Forensic Analytics Platform (Sidebar Navigation, 7 Workspaces)
├── main.py                 # Pipeline: CSV feature extraction, RF training & ONNX export
├── ml_server.py            # FastAPI backend server for real-time model inference (port 8000)
├── paternity_model.onnx    # Exported ONNX model for browser client & server inference
├── paternity_model.pkl     # Serialized Scikit-Learn Random Forest model
├── feature_cols.pkl        # Serialized feature list for exact inference alignment
├── model_metrics.json      # Accuracy, ROC-AUC, precision, recall & confusion matrix
├── allele_frequencies.json # CODIS allele population database (Caucasian, African American, etc.)
├── paternity.csv           # Benchmark paternity dataset (20 CODIS core loci triads)
├── data.py                 # Data preprocessing and synthetic triad generator
└── requirements.txt        # Python dependency manifest
```

---

## 🧪 ML Pipeline

### Training Workflow (`main.py`)

```python
# 1. Load CSV data (paternity.csv)
# 2. Extract 180 features:
#    - compat_<locus> (0/1): Mendelian compatibility per CODIS locus
#    - total_compatible_loci: Count of compatible loci
#    - combined_lr: Product of all locus PIs
#    - log_combined_lr: log₁₀(CPI)
# 3. Train Random Forest classifier
# 4. Export to ONNX format (wasm execution provider)
# 5. Generate ml_server.py for REST API
```

### Feature Engineering

**Input Features (22 total):**
- `compat_csF1po` through `compat_d22S1045` (20 binary compatibility flags)
- `total_compatible_loci` (integer, 0-20)
- `log_combined_lr` (float, log₁₀ scale)

**Output Class:**
- Paternity classification (binary or multi-class based on model)

---

## 🔬 Forensic Analysis Engine

### Mendelian Compatibility Rules

```javascript
// Pseudo-code from model.html
function evaluateLocus(locus, mother, child, allegedFather):
  // Case 1: Mother has c1 only → c2 is obligate paternal
  if (c1InM && !c2InM):
    obligate = c2
    if (c2InF): 
      status = 'Match'
      PI = (0.5 or 1.0) / alleleFrequency(locus, c2)
    else:
      check mutation → apply reduced PI based on mutation rate

  // Case 2: Maternal exclusion
  else if (!c1InM && !c2InM):
    status = 'Exclusion'
    PI = 0.001
```

### Mutation Detection

| Condition | Interpretation |
|-----------|----------------|
| Δ = 1 repeat unit | Single-step mutation (e.g., 12 → 13) |
| Δ = 0.2 repeat units | Two-allele partial mutation |
| Δ > 1 repeat unit | Exclusion (no mutation) |

**Mutation Rate Database:**
```python
MUTATION_RATE = {
    "CSF1PO": 0.0016,
    "D3S1358": 0.0012,
    "D5S818": 0.0023,
    # ... (all 20 CODIS loci)
}
```

### Likelihood Ratio Calculation

**Per-Locus PI:**
- Match (no mutation): `(0.5 or 1.0) / allele_freq`
- Match with mutation: `(mutation_rate × 0.5) / allele_freq`
- Exclusion: `0.001`

**Combined PI (CPI):**
```
CPI = Π(locus_PI_for_all_20_loci)
```

**Posterior Probability:**
```
PP = (CPI × prior) / [(CPI × prior) + (1 - prior)] × 100%
```

---

## 📈 ML Model Performance

### Validation Metrics (from model.html UI)

| Metric | Value |
|--------|-------|
| **Random Forest ROC-AUC** | To be filled with actual training results |
| **Decision Tree ROC-AUC** | Baseline comparison |
| **Weighted F1-Score** | Harmonic mean of precision/recall |
| **Global Accuracy** | Overall classification accuracy |

### Model Interpretation

The Random Forest classifier:
- Accepts 22-dimensional feature vector (20 compatibilities + 2 summary stats)
- Outputs probability distribution over classes
- Provides inference confidence scores

---

## 🖥️ User Interface & Workspaces (`model.html` & `index.html`)

The GENETRACE platform features a cohesive dark scientific aesthetic (`#080D1B` background, `#111A2D` panels, `#55EADB` cyan accents) paired with a persistent collapsible sidebar and 7 dedicated functional workspaces:

### 1. Unified Navigation Architecture
- **Persistent Collapsible Sidebar**: Allows instant switching between all modules without page reloads.
- **Top Utility Header**: Real-time ML engine status indicator (green pulse when backend active, fallback to local core), active subject breadcrumb, demo cycle trigger, new case dialog, CSV importer, and direct print shortcut.
- **Cross-App Integration**: Seamless bi-directional navigation between `index.html` (3D educational guide) and `model.html` (forensic analytics core).

### 2. Workspaces Overview

1. **Dashboard**
   - System-wide overview KPI cards: Total Cases, Active Ingestions, Confirmed Inclusions, and Ruled Exclusions.
   - Recent cases table with live status badges and one-click case activation.
   - Forensic Activity stream logging real-time ingestion, mutation detection, and report generation events.
   - Quick Action hub for loading demo triads, importing CSVs, and creating cases.

2. **Case Management**
   - Comprehensive case repository with live search (Case ID, analyst, cohort) and status filters (*All*, *Completed*, *Review Required*, *Draft*).
   - Case metadata editor with demographic configuration and population cohort selector.

3. **DNA Triad Analysis**
   - 20 CODIS core loci genotype editor (Mother, Child, Alleged Father).
   - Real-time validation and Mendelian allele extraction.
   - Quick preset generation buttons: *Standard Inclusion*, *Single Mutation (Step 1)*, and *Multi-Locus Exclusion*.

4. **Forensic Analytics (Modular Sub-Views)**
   - **Analytics Overview**: KPI cards showing Evaluated Loci, Compatible Loci, Mutation Count, Exclusion Count, and log₁₀(CPI).
   - **Locus Evidence**: Interactive Chart.js log₁₀(Paternity Index) visualization with match, mutation, and exclusion threshold markers, alongside a widened, responsive 20-locus tabular inspection matrix.
   - **Statistical Evidence**: Mathematical formulation breakdown, neutral prior vs. posterior probability calculations, competing forensic hypotheses ($H_1$ vs. $H_2$), and population frequency assumptions.

5. **ML Validation Workspace**
   - Dedicated evaluation dashboard separating machine learning supplementary predictions from classical statistical evidence.
   - Interactive 2×2 Confusion Matrix (True Positive, True Negative, False Positive, False Negative).
   - Validation performance metrics (Accuracy, ROC-AUC, Precision, Recall, F1-Score).
   - Top predictive feature importance ranking (e.g., `total_compatible_loci`, `log_combined_lr`).

6. **Clinical Reports & Isolated Single-Page PDF Export**
   - **Preview Full Report**: Modal preview of the official clinical document.
   - **Isolated Single-Page Print Pipeline**: Custom print document renderer using an isolated, zero-dependency `iframe` (`srcdoc` pipeline) to ensure 100% clean output. Eliminates application UI bleed, background buttons, and blank pages.
   - **Classic Formal Clinical Format**:
     - Letterhead: `CORE STR ANALYTICS LABS // FORENSIC GENETICS DIVISION`
     - Chain-of-Custody Certification block with Analyst Signature, Ingestion ID, and Engine.
     - Final Parentage Classification verdict banner.
     - Complete 20 CODIS Core Loci Formal Summary Matrix.
     - Methodology & Statistical Foundation notes and Certified Analyst Sign-off.
   - Configured with `@page { size: A4 portrait; margin: 0.6cm 0.7cm; }` for a strict 1-page fit.
   - Data export options: Full CSV and JSON exports.

7. **Settings & Database**
   - Backend ML server endpoint configuration (`http://127.0.0.1:8000/predict`).
   - Population allele database reference tables.
   - Local storage management and session reset controls.

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.15+
- pip, node.js (for optional UI enhancements)
- Browser with modern JavaScript support (Chrome/Firefox/Edge)

### Installation Steps

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install Python dependencies
pip install fastapi uvicorn pydantic onnxruntime numpy pandas scikit-learn

# 3. Prepare training data
cp sample_data/paternity.csv data/paternity.csv

# 4. Train model and export ONNX
python main.py

# 5. Start ML server
python ml_server.py

# 6. Open forensic interface
open model.html  # macOS | start model.html.cmd (Windows)
```

### Development Mode

Edit `model.html` directly — it's a self-contained HTML file that:
- Loads Chart.js, Tailwind CSS, and ONNX Runtime from CDN
- Can connect to local ML server at `http://localhost:8000`
- Falls back to local forensic engine if backend is unavailable

---

## 📊 Data Format

### Input CSV Schema (paternity.csv)

| Column | Type | Description |
|--------|------|-------------|
| Locus | string | CODIS locus name |
| M_Allele1 | integer | Mother's first allele |
| M_Allele2 | integer | Mother's second allele |
| C_Allele1 | integer | Child's first allele |
| C_Allele2 | integer | Child's second allele |
| AF_Allele1 | integer | Alleged Father's first allele |
| AF_Allele2 | integer | Alleged Father's second allele |

**Example Row:**
```csv
CSF1PO,10,11,11,12,12,12
D3S1358,12,14,14,16,14,17
FGA,21,24,24,26,26,27
```

---

## 🔐 Forensic Standards & Compliance

### Validation Basis

- **Butler JM.** *Forensic DNA Typing: Biology, Technology, and Genetics of STR Markers*. 3rd ed. Academic Press, 2020.
- **NIST/STRBase** allele frequency databases
- **CODIS Core Loci** as defined by FBI (20 standards)

### Report Standards

Generated reports include:
- Chain-of-custody information block
- Analyst certification signature area
- Case metadata (population group, analyst name)
- Statistical evidence summary
- AI-generated inference disclaimer

---

## 🚧 Limitations

1. **Population Bias**: Default allele frequencies are Caucasian-based; other populations may require recalibration
2. **Single Child per Case**: Triad analysis assumes one child per case
3. **No Related Alleged Fathers**: Does not handle scenarios with multiple related candidates
4. **ML Model Interpretation**: Random Forest predictions are supplementary to validated STR engine

---

## 🔮 Future Enhancements

- [ ] Multi-child support (pedigree analysis)
- [ ] Expanded population databases (African American, Hispanic, Asian)
- [ ] Y-STR and mitochondrial DNA analysis modules
- [ ] Relatedness testing (full-sibling, half-sibling, uncle-nephew)
- [ ] Real-time ML backend deployment on cloud infrastructure
- [ ] Bulk case batch processing
- [ ] Electronic signature integration for report validation

---

## 📄 License

MIT License — See LICENSE file for details.

> **Note**: This software is for educational and research purposes. For clinical or legal use, validate against jurisdiction-specific forensic standards and obtain appropriate accreditation.

---

## 🆘 Support & Issues

For technical issues, feature requests, or dataset questions:
- Email me on vanthex.dev@gmail.com
- Review ML pipeline code in `main.py` and `ml_engine.py`
- Inspect forensic logic in `model.html` JavaScript section

---

**Built with precision for the future of forensic genomics.**
