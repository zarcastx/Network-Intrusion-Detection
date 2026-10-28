"""CSET312 CICIDS2017 offline CSV intrusion-classification demonstration."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='CSET312 Network Intrusion Detector', layout='wide')
st.title('AI-Powered Network Intrusion Detection')
st.caption('Offline CICIDS2017-style network-flow classification. Not live packet monitoring.')

model_path = Path(__file__).with_name('selected_model.joblib')
if not model_path.exists():
    st.error('Place selected_model.joblib next to app.py.'); st.stop()
try:
    artifact = joblib.load(model_path)
    pipe = artifact.get('model', artifact.get('pipeline'))
    features = artifact.get('feature_names', artifact.get('features'))
    name = artifact.get('model_name', artifact.get('selected_name', 'Saved classifier'))
    if pipe is None or not features:
        raise ValueError('Model artifact lacks model/pipeline or feature list')
except Exception as exc:
    st.error(f'Could not load model: {exc}'); st.stop()

st.info(f'Model: {name} • Expected numeric features: {len(features)}')
file = st.file_uploader('Upload a CSV containing the trained model features', type=['csv'])
if file is not None:
    try:
        raw = pd.read_csv(file)
        raw.columns = raw.columns.str.strip()
        if raw.empty:
            st.warning('The uploaded CSV contains no data rows.'); st.stop()
        missing = [f for f in features if f not in raw.columns]
        if missing:
            st.error(f'Missing {len(missing)} features, such as: {missing[:10]}'); st.stop()
        X = raw.loc[:,features].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan)
        y_pred = pipe.predict(X)
        result = raw.copy()
        result['Predicted_Status'] = np.where(y_pred == 1, 'ATTACK', 'BENIGN')
        c1,c2,c3 = st.columns(3)
        c1.metric('Records processed', len(result))
        c2.metric('Predicted attacks', int((y_pred == 1).sum()))
        c3.metric('Predicted benign', int((y_pred == 0).sum()))
        st.dataframe(result[['Predicted_Status']].reset_index(drop=True), use_container_width=True)
        st.download_button('Download prediction results', result.to_csv(index=False).encode('utf-8'), 'intrusion_predictions.csv', 'text/csv')
        st.caption('Predictions are estimates; verify suspicious flows with additional security evidence.')
    except Exception as exc:
        st.error(f'Could not classify this CSV: {exc}')
