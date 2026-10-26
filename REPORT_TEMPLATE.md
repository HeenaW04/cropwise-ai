# CropWise AI: A Comparative Machine Learning Approach for Intelligent Crop Recommendation

**Mini Project Report — Machine Learning**  
Student: [Your name]  | Roll No.: [Your roll number]  | Class: B.Tech CSE, Semester V  
Faculty Guide: [Guide name]  | Institute: [College name]  | Academic Year: 2026–27

> Report preparation note: replace every bracketed placeholder after running the experiments. Insert the actual metric table and plots from `results/`. Do not submit invented scores. Adjust page breaks/spacing to meet the department's 10-page format.

## 1. Introduction

Agriculture depends on interactions among soil nutrients and environmental conditions. Selecting a crop that is compatible with these conditions is a complex decision, and machine learning can provide a data-driven starting point. This project develops CropWise AI, a multiclass classification system that predicts a crop label from seven input variables: nitrogen (N), phosphorus (P), potassium (K), temperature, relative humidity, soil pH and rainfall.

The project compares six classical supervised learning algorithms under a consistent train-test split and reports accuracy, macro-averaged precision, recall and F1-score. It also includes exploratory data analysis, preprocessing, cross-validation, confusion-matrix analysis, and a Streamlit user interface. The main educational goal is a reproducible comparison and deployment workflow, rather than a claim that a model can guarantee agricultural success.

**Objectives:** (1) inspect and clean the dataset; (2) split data without class imbalance; (3) train at least four classifiers; (4) compare metrics and visualizations; (5) deploy an interactive prediction interface; and (6) document limitations and reproducibility.

## 2. Literature Review

### 2.1 Senapaty, Ray and Padhy (2024)
The paper “A Decision Support System for Crop Recommendation Using Machine Learning Classification Algorithms” presents crop recommendation as a supervised learning and decision-support problem. It provides relevant context for comparing classification algorithms and building a usable recommendation workflow. For CropWise AI, the key lesson is to compare algorithms on the same task and evaluate the output systematically. Unlike a field-validated agronomy service, this mini-project is a benchmark prototype. DOI: https://doi.org/10.3390/agriculture14081256.

### 2.2 Shams, Gamel and Talaat (2024)
“Enhancing crop recommendation systems with explainable artificial intelligence: a study on agricultural decision-making” explores explainable AI for crop recommendation. The work highlights that strong predictive performance alone may not make recommendations transparent to users. CropWise AI includes interpretable benchmark outputs and feature-level EDA; a future extension could add SHAP explanations. DOI: https://doi.org/10.1007/s00521-023-09391-2.

### 2.3 Srinivasu and Ijaz (2024)
“XAI-driven model for crop recommender system for use in precision agriculture” examines explainability in precision-agriculture crop recommendation. It motivates future work that can explain which environmental and soil inputs influence a prediction. The current implementation focuses on reproducible baseline comparisons and a web interface, so explainability methods beyond standard metrics are identified as future work. DOI: https://doi.org/10.1111/coin.12629.

### 2.4 Machine learning based recommendation of agricultural and horticultural crop farming in India (2024)
This Heliyon paper compares SVM, XGBoost, Random Forest, KNN and Decision Tree models using NPK, pH and climate variables. It reports that performance differs across crop groups and that XGBoost performs strongly in its experiments. This supports the need to benchmark multiple models instead of assuming one algorithm will always be best. Its datasets and experimental design are not identical to the current mini-project, so its published scores must not be treated as directly comparable to ours. DOI: https://doi.org/10.1016/j.heliyon.2024.e25112.

### 2.5 Shastri et al. (2025)
“Advancing crop recommendation system with supervised machine learning and explainable artificial intelligence,” published in Scientific Reports, studies supervised crop recommendation and explainability using the common 2,200-record, 22-class dataset. It reinforces the relevance of data preparation, model comparison and interpretation. CropWise AI uses the benchmark dataset for educational reproducibility but must acknowledge that benchmark performance does not establish local agricultural validity. DOI: https://doi.org/10.1038/s41598-025-07003-8.

### 2.6 Comparative summary
Across these studies, recurring themes include soil/climate-based classification, comparison of multiple algorithms, and the need for explainability and practical interfaces. Some studies use additional data, different crop groups, advanced ensembles or region-specific observations. Their numerical results should not be directly ranked against this project because data versions, splits, preprocessing and evaluation protocols differ. CropWise AI contributes a transparent, beginner-friendly and reproducible baseline experiment with exported metrics and a deployed interface.

## 3. Methodology

The workflow consists of dataset acquisition, schema validation, exploratory data analysis, cleaning, stratified splitting, model training, performance evaluation, saving the selected pipeline and deployment through Streamlit. The target variable is `label`, representing one of 22 crop classes. The seven predictor variables are N, P, K, temperature, humidity, pH and rainfall.

The dataset is split into 80% training and 20% test partitions with stratification and random seed 42. Stratification maintains the approximate class proportions in both partitions. Preprocessing is included inside scikit-learn pipelines so imputation and scaling are fitted on training data rather than the held-out test data.

## 4. Data Cleaning and Preprocessing

The CSV schema and column names are checked. Numeric columns are converted to numeric types; invalid numeric values become missing values. Rows without target labels and exact duplicate records are removed. Numeric missing values, if any, are imputed with the median within each model pipeline. The class distribution is visualized and numerical feature correlations are plotted.

**Measured cleaning summary:** Original rows = [fill from `results/run_summary.json`]; retained rows = [fill]; removed rows = [fill]; number of features = 7; number of classes = [fill]. Report missing-value counts before imputation and duplicate counts from the executed dataset inspection. Do not claim the dataset contains missing values or duplicates unless your run confirms it.

## 5. Models and Evaluation Metrics

Six models are evaluated: Decision Tree, Random Forest, Support Vector Machine with RBF kernel, K-Nearest Neighbors, Logistic Regression and Extra Trees. Decision Tree and Random Forest represent tree-based learning; SVM, KNN and Logistic Regression use standardized inputs through pipelines; Extra Trees provides an additional randomized tree ensemble baseline.

Accuracy is the fraction of correctly classified test examples. Macro precision, macro recall and macro F1 calculate each metric separately for every class and then average equally across classes, avoiding domination by frequent classes. Five-fold stratified cross-validation macro F1 is computed on the training partition. Fit time is recorded as a practical efficiency measure.

## 6. Results and Discussion

Insert `results/model_comparison.csv` as **Table 1: Measured model comparison**. Insert `results/figures/model_accuracy_comparison.png`, `model_f1_comparison.png`, `class_distribution.png`, `correlation_heatmap.png`, and `confusion_matrix_best_model.png` as Figures 1–5. Use the model outputs generated by your own run.

**Measured best model:** [fill from `results/run_summary.json`]. **Test accuracy:** [fill]. **Test macro-F1:** [fill]. Discuss whether the model with the highest accuracy also has the highest macro-F1, which crop classes are confused in the confusion matrix, and whether the cross-validation scores are consistent with the held-out score. If scores are unexpectedly close to 100%, discuss the limitations of the small, structured benchmark and the need for external validation.

## 7. User Interface Development and Deployment

The interface is built with Streamlit. Users enter N, P, K, temperature, humidity, pH and rainfall through sidebar controls and click “Recommend crop.” The application loads the saved scikit-learn model, returns the predicted class, and displays the top three model outputs when probability estimates are supported. The benchmark metrics and macro-F1 comparison figure are also displayed when the result files are available.

For deployment, the source code is stored in GitHub and the app can be hosted on Streamlit Community Cloud. The app requires `app.py`, `requirements.txt`, and `models/best_model.joblib`; therefore, the trained model artifact must be present in the repository or made available through a documented model-download process. The deployment was tested at: [insert URL]. Devices/browsers tested: [insert].

## 8. Limitations and Future Scope

The dataset is small and represents a simplified mapping from seven features to crop labels. Its provenance and geographic coverage may not represent every region or season. The model does not directly account for soil texture, crop variety, planting calendar, irrigation, pests, market prices or measured yield. Predicted class probabilities may not be calibrated. Future work could use locally collected soil-test and weather data, region- and season-specific validation, a top-k recommendation interface, SHAP explanations, model calibration, and agronomist review. Real-world use should be validated before advising farmers.

## 9. Conclusion

CropWise AI demonstrates an end-to-end supervised machine-learning workflow for crop recommendation: data validation and cleaning, EDA, stratified train-test splitting, comparison of multiple classifiers, macro-averaged performance analysis, visualization, model serialization and web deployment. The final conclusion should be based on the measured results generated by the training script. The system is a learning prototype and decision-support demonstration, not a substitute for expert agricultural advice or field trials.

## References

1. Senapaty, M. K., Ray, A., & Padhy, N. (2024). A Decision Support System for Crop Recommendation Using Machine Learning Classification Algorithms. *Agriculture, 14*(8), 1256. https://doi.org/10.3390/agriculture14081256
2. Shams, M. Y., Gamel, S. A., & Talaat, F. M. (2024). Enhancing crop recommendation systems with explainable artificial intelligence: a study on agricultural decision-making. *Neural Computing and Applications, 36*, 5695–5714. https://doi.org/10.1007/s00521-023-09391-2
3. Srinivasu, P. N., & Ijaz, M. F. (2024). XAI-driven model for crop recommender system for use in precision agriculture. *Computational Intelligence, 40*(1), e12629. https://doi.org/10.1111/coin.12629
4. Machine learning based recommendation of agricultural and horticultural crop farming in India under the regime of NPK, soil pH and three climatic variables. (2024). *Heliyon*. https://doi.org/10.1016/j.heliyon.2024.e25112
5. Shastri, S., Kumar, S., Mansotra, V., et al. (2025). Advancing crop recommendation system with supervised machine learning and explainable artificial intelligence. *Scientific Reports, 15*, 25498. https://doi.org/10.1038/s41598-025-07003-8
6. Dataset: Crop Recommendation Dataset, 2,200 records and 22 classes. Public dataset page: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset. Cite the exact version and access date used in your submission.
