# Colab cells in order

### Cell 1 — Download dataset
```python
from pathlib import Path
import pandas as pd
Path('data').mkdir(exist_ok=True)
url='https://raw.githubusercontent.com/niteshwayakar/Crop-Recommendation/main/Crop_recommendation.csv'
df=pd.read_csv(url)
df.to_csv('data/Crop_recommendation.csv',index=False)
print('Shape:',df.shape)
print(df.head())
print(df.isna().sum())
print('Duplicates:',df.duplicated().sum())
print(df['label'].value_counts())
```

### Cell 2 — Upload the project training script
Upload `src/train_model.py` into a folder named `src` in Colab. Keep the dataset at `data/Crop_recommendation.csv` relative to the project root. Alternatively upload the full project ZIP and unzip it.

### Cell 3 — Install and train
```python
!pip -q install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
!python src/train_model.py
```

### Cell 4 — Read measured results
```python
import pandas as pd, json
from IPython.display import display, Image
display(pd.read_csv('results/model_comparison.csv'))
print(open('results/run_summary.json').read())
for f in ['class_distribution.png','correlation_heatmap.png','model_accuracy_comparison.png','model_f1_comparison.png','confusion_matrix_best_model.png']:
    display(Image('results/figures/'+f))
```

### Cell 5 — Download artifacts
Zip the project including `models/` and `results/`, then download the ZIP from Colab. Before deployment, ensure the saved model exists.
