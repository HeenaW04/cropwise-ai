from pathlib import Path
import joblib, streamlit as st
import pandas as pd
ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/'models'/'best_model.joblib'
FEATURES=['n','p','k','temperature','humidity','ph','rainfall']
st.set_page_config(page_title='CropWise AI',page_icon='🌱',layout='wide')
st.title('🌱 CropWise AI')
st.caption('Machine-learning-assisted crop recommendation from soil nutrients and climate conditions')
st.warning('Educational decision-support prototype only. The public benchmark dataset is limited and does not replace local agricultural expertise, soil testing, or region-specific advice.')
if not MODEL_PATH.exists():
    st.error('Trained model not found. Run src/train_model.py first; see README.md for Google Colab steps.'); st.stop()
model=joblib.load(MODEL_PATH)
with st.sidebar:
    st.header('Field conditions')
    n=st.number_input('Nitrogen (N)',min_value=0.0,max_value=2000.0,value=90.0,step=1.0)
    p=st.number_input('Phosphorus (P)',min_value=0.0,max_value=2000.0,value=42.0,step=1.0)
    k=st.number_input('Potassium (K)',min_value=0.0,max_value=2000.0,value=43.0,step=1.0)
    temperature=st.number_input('Temperature (°C)',min_value=-10.0,max_value=60.0,value=25.0,step=0.5)
    humidity=st.number_input('Humidity (%)',min_value=0.0,max_value=100.0,value=80.0,step=1.0)
    ph=st.number_input('Soil pH',min_value=0.0,max_value=14.0,value=6.5,step=0.1)
    rainfall=st.number_input('Rainfall (mm)',min_value=0.0,max_value=2000.0,value=200.0,step=1.0)
    submitted=st.button('Recommend crop',type='primary',use_container_width=True)
if submitted:
    row=pd.DataFrame([[n,p,k,temperature,humidity,ph,rainfall]],columns=FEATURES)
    prediction=model.predict(row)[0]
    st.subheader('Recommendation')
    st.success(f'Predicted crop: **{str(prediction).title()}**')
    st.write('This is the model’s predicted class based on the entered values, not a guarantee of crop suitability or yield.')
    if hasattr(model,'predict_proba'):
        probs=model.predict_proba(row)[0]
        classes=list(model.classes_)
        top=sorted(zip(classes,probs),key=lambda x:x[1],reverse=True)[:3]
        st.subheader('Top model outputs')
        st.dataframe(pd.DataFrame([{'Crop':str(c).title(),'Model probability':float(v)} for c,v in top]),use_container_width=True,hide_index=True)
        st.caption('Probabilities are model scores and may not be calibrated probabilities of real-world success.')
metrics_path=ROOT/'results'/'model_comparison.csv'
if metrics_path.exists():
    st.divider(); st.subheader('Benchmark results')
    metrics=pd.read_csv(metrics_path)
    st.dataframe(metrics.style.format({c:'{:.3f}' for c in metrics.columns if c!='Model'}),use_container_width=True,hide_index=True)
    fig_path=ROOT/'results'/'figures'/'model_f1_comparison.png'
    if fig_path.exists(): st.image(str(fig_path),caption='Measured macro F1-score on the held-out test set')
else:
    st.info('After training, this page will display the measured model comparison table and figure.')
st.subheader('Input feature guide')
st.write('N, P and K are nutrient measurements as represented in the dataset; temperature is in °C; humidity is relative humidity (%); pH is soil acidity/alkalinity; rainfall is in mm.')
