
# Sex-specific Machine Learning Models for Predicting PDAC Survival

This repository contains code used in the study titled:  
Gap-App: A Sex-Distinct AI-Based Predictor for Pancreatic Ductal Adenocarcinoma Survival as A Web Application Open to Patients and Physicians
submitted to Cancer Letters.

## Overview

We developed and evaluated sex-specific machine learning (ML) models to predict 3-year overall survival in patients with pancreatic ductal adenocarcinoma (PDAC) using transcriptomic data. The code in this repository supports the preprocessing of raw data, model training, performance evaluation, and figure generation for the manuscript.

## Repository Structure

```
├── data_preprocessing with hazard ratio and differential gene expression analysis/
│   ├── coxph_HR_calculation_TCGA.R
│   └── DGE_analysis_TCGA_GTEX.R
├── model_training with cross-validation metrics and hyperparameter tuning/
│   └── Model_optimization_and_evaluation.py
├── Feature selection from initial models/
│   └── Feature_Selection.py
├── evaluation/
│   ├── ML_model_evaluation.py
│   ├── Calibration.py
│   └── Final_ML_models.py
├── figures/
│   └── ROC_curve.py
├── requirements.txt
└── README.md
```

## Data Sources

The models were developed and validated using publicly available datasets:

- **Training dataset:** TCGA PAAD cohort via UCSC Xena (https://xenabrowser.net/) and GTEx normal pancreas samples via Toil Hub (https://toil.xenahubs.net).
- **Validation dataset:** GEO accession [GSE79668](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE79668) (BioProject [PRJNA316673](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA316673)), raw FASTQ files available at [SRP072492](https://www.ncbi.nlm.nih.gov/sra/SRP072492).

Detailed acquisition and preprocessing steps are described in the manuscript's Methods section.

## Instructions

### 1. Setup

Install required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data and Initial Gene Filtration

Run preprocessing scripts for TCGA/GTEx and GEO datasets along with hazard ratio and differential gene expression analysis:

```bash
Rscript coxph_HR_calculation_TCGA.R
Rscript DGE_analysis_TCGA_GTEX.R
```

### 3. Hyperparameter tuning and cross-validation metrics for initial models

```bash
python Model_optimization_and_evaluation.py
```

### 4. Feature selection from initial models

```bash
python Feature_Selection.py
```

### 5. Hyperparameter tuning and cross-validation metrics for refined models

```bash
python Model_optimization_and_evaluation.py
```

### 6. Evaluate Models and Generate Figures

```bash
python ML_model_evaluation.py
python Calibration.py
python Final_ML_models.py
python ROC_curve.py
```

## Notes

- If the automatic data import of the datasets in the R script does not work, please use the provided datasets to import it in R.
- All gene expression values were normalized to log₂(FPKM + 1).
- Sex-specific models were trained separately on male and female subsets.
- The deployed version of the model (accessible at https://www.gap-app.org/) was not used to generate results in this manuscript and is not included here.
- The final trained models (in `.pkl` format) can be regenerated using the training scripts provided in this repository and are also available upon reasonable request after publication or pending student project completion.

## License

This web application and its underlying code are licensed under the **MIT License**. It is freely available for academic, clinical, and public use. Researchers, clinicians, patients with pancreatic ductal adenocarcinoma (PDAC), and their caregivers are welcome to use this tool to support education, analysis, and decision-making.

Please cite the associated publication when using or referencing this work.

The full license text is provided below:

```
MIT License

Copyright (c) 2024 JingYuan Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
