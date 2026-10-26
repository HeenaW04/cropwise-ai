# CropWise AI — Crop Recommendation using Machine Learning

A beginner-friendly multiclass classification project for a Machine Learning mini-project. It compares six classifiers, exports measured metrics and figures, saves a trained model, and provides a Streamlit interface.

## Dataset
Use the public 2,200-row Crop Recommendation benchmark CSV with columns `N, P, K, temperature, humidity, ph, rainfall, label`. Dataset source page: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset . A public mirror for convenient Colab download is: https://raw.githubusercontent.com/niteshwayakar/Crop-Recommendation/main/Crop_recommendation.csv . Check the dataset's license/provenance before redistribution. The CSV is not bundled in this repository; download it yourself into `data/Crop_recommendation.csv`.

## Google Colab — step by step
1. Download this project ZIP and extract it on your computer.
2. Open https://colab.research.google.com and create a new notebook.
3. Upload `Crop_recommendation.csv` from the dataset page, or run the cell below to download the public mirror:

```python
from pathlib import Path
import pandas as pd
Path('data').mkdir(exist_ok=True)
url = 'https://raw.githubusercontent.com/niteshwayakar/Crop-Recommendation/main/Crop_recommendation.csv'
df = pd.read_csv(url)
df.to_csv('data/Crop_recommendation.csv', index=False)
print(df.shape)
df.head()
```

4. Upload `src/train_model.py` to Colab or upload/extract the whole project ZIP. If using the whole project in Colab, change directory to the project root.
5. Install libraries:

```python
!pip install -r requirements.txt
```

6. Run the training script from the project root:

```python
!python src/train_model.py
```

7. Inspect `results/model_comparison.csv`, `results/run_summary.json`, `results/best_model_classification_report.csv`, and images in `results/figures/`. These are your actual experimental results. Do not copy any example numbers from papers as your own results.
8. Download the project folder/outputs from Colab. Ensure `models/best_model.joblib`, `models/feature_names.joblib`, and `results/` are available before running the app.

## Run the Streamlit app locally
Install Python 3.10+ and Git. Open a terminal in the project folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## GitHub and Streamlit Community Cloud
1. Create a new public GitHub repository named `cropwise-ai`.
2. Upload project files, but do not upload private credentials or unrelated files. The small trained model can be committed if your course rules and repository-size limits permit it; otherwise retrain during setup and use an artifact hosting approach.
3. Confirm the root contains `app.py` and `requirements.txt`, and the trained model is at `models/best_model.joblib`.
4. Visit https://share.streamlit.io/ or Streamlit Community Cloud, sign in with GitHub, choose the repository and branch, set main file to `app.py`, and deploy.
5. Test the public URL in a private/incognito window. If the model file is missing, the app will show an error; include the model artifact or arrange a reliable model download/training step.

## Methodology and leakage prevention
- Exact duplicate rows are removed; rows without target labels are removed. Numeric missing values are handled inside each estimator pipeline by median imputation.
- Use a stratified 80:20 train-test split and fixed random seed 42.
- Scaling is fitted inside pipelines for SVM, KNN and Logistic Regression, avoiding leakage from the test set.
- Compare Decision Tree, Random Forest, RBF-SVM, KNN, Logistic Regression and Extra Trees.
- Report test accuracy, macro precision, macro recall, macro F1, 5-fold training-set cross-validation macro F1, and fit time.
- Select the displayed best model by test macro-F1 for a student demonstration. For a stricter final scientific estimate, choose models using training-only cross-validation and evaluate the chosen model once on the held-out test set. Mention this distinction in the report.

## Important caveats
This is a benchmark demonstration, not a validated agronomy tool. The widely used dataset is small and may have synthetic/aggregated origins, limited geographic context, and possible distribution shift. Never claim real yield improvement or guaranteed recommendations without field validation. The dataset's features are not enough to encode every local factor (season, soil type, cultivar, irrigation, pests, prices, local planting calendar).
