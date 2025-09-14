# MMM Mediation Pipeline

## Overview
This repository implements a **Marketing Mix Modeling (MMM) pipeline** with a **two-stage mediation approach**:

1. **Stage 1**: Predict Google ad spend from social media channels (mediator).  
2. **Stage 2**: Predict revenue from the Google mediator + control variables (price, promotions, emails, SMS, social followers).

The pipeline handles:
- Weekly seasonality and trends  
- Zero-spend periods  
- Adstocking and log transformations  
- Lag features  
- Ridge regression with cross-validated alpha  
- Out-of-sample predictions  
- Sensitivity analysis for Average Price and Promotions  

---

## Repo Structure

```

MMM\_Mediation/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── data/
│   └── data.csv                  # Weekly dataset with revenue, spends, and controls
├── src/
│   └── MMM\_Mediation\_Assessment.py  # Main pipeline script
├── outputs/                      # Auto-generated outputs (plots, CSVs)
├── models/                       # Saved Ridge models and scalers
└── reports/                      # Automated write-ups and sensitivity analysis

````

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/SAANIYASALAM/MMM_Mediation.git
cd MMM_Mediation
````

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
```

3. **Activate the environment**

* **Windows**

```bash
venv\Scripts\activate
```

* **Mac/Linux**

```bash
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python src/MMM_Mediation_Assessment.py
```

* This will automatically:

  * Load your `data/data.csv`
  * Engineer features (adstock, log, lags, Fourier terms, trends)
  * Run Stage 1 (social → Google) and Stage 2 (revenue ← Google + controls)
  * Save models in `models/`
  * Save outputs (plots, CSVs) in `outputs/`
  * Generate reports in `reports/`

---

## Notes

* Configurable parameters are at the top of the script, e.g., `ADSTOCK_HALFLIFE_WEEKS`, `FOURIER_ORDER`, `LAG_WEEKS`.
* The script produces **out-of-sample predictions** for both stages to prevent look-ahead bias.
* Elasticity estimates and mediator sensitivity are saved in the `reports/` folder.
* If you update the code or `requirements.txt`, commit and push changes to keep GitHub repo up-to-date.

---

## Contact

* **Author**: Saaniya Salam
* **Email**: [saaniyasalam@gmail.com](mailto:saaniyasalam@gmail.com)

