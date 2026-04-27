# Coming soon

# CAPPER

**CAPPER** (**Cap**illary / **Per**icyte Quantification Tool) is a Python-based image analysis tool designed for automatic quantification of capillary density and pericyte coverage in microscopy images.

---

## Abstract

The vascular system ensures sufficient blood supply and tissue homeostasis and consists of different cell types. Endothelial cells represent the structural backbone of blood vessels and are accompanied by murals cells, specifically pericytes in the microcirculation as well as vascular smooth muscle cells (vSMC) along the larger vessels (arteries, arterioles and veins). Distinguishing these different cell types in immunohistochemical stainings presents a challenge due to their close proximity and the unreliable marker distribution of mural cells. Furthermore, manual quantification of capillaries and pericytes is highly examiner-dependent, hindering inter-examiner and inter-laboratory comparisons. To address these issues, we developed an automated algorithm-based analysis software designed to standardize quantification of vascular structures in immunohistochemical images, named CAPPER (**Cap**illary / **Per**icyte Quantification Tool). Through the implementation of adaptive thresholding, morphological operations, and domain-specific knowledge, CAPPER excels in the quantification of capillary densities and pericyte coverage in different organs (brain, heart, muscle, and kidney), species (mouse and pig) as well as a plethora of disease states.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ariogato/capper.git
cd capper
```
### 2. (Optional) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the main script
```bash
python main.py
```

