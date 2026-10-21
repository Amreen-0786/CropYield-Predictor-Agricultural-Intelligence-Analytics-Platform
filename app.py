from pathlib import Path
import joblib, pandas as pd, numpy as np, streamlit as st, plotly.express as px
st.set_page_config(page_title="CropYield Predictor",page_icon="🌾",layout="wide")
st.title("🌾 CropYield Predictor")
st.caption("Agricultural intelligence: yield prediction, EDA, and model comparison")
model_path=Path("models/cropyield_pipeline.joblib")
if not model_path.exists(): st.error("Model file missing. Run python train.py first."); st.stop()
model=joblib.load(model_path)
data_path=Path("data/sample/crop_yield_demo.csv")
df=pd.read_csv(data_path) if data_path.exists() else pd.DataFrame()
tab1,tab2,tab3=st.tabs(["EDA Insights","Yield Prediction","Model Comparison"])
with tab1:
    st.subheader("Dataset overview")
    if not df.empty:
        c1,c2,c3=st.columns(3); c1.metric("Rows",len(df)); c2.metric("Features",df.shape[1]-1); c3.metric("Missing cells",int(df.isna().sum().sum()))
        st.plotly_chart(px.histogram(df,x="Yield_ton_ha",title="Yield Distribution"),use_container_width=True)
        st.plotly_chart(px.box(df,x="Crop",y="Yield_ton_ha",title="Yield by Crop"),use_container_width=True)
        st.plotly_chart(px.scatter(df,x="Rainfall_mm",y="Yield_ton_ha",color="Crop",title="Rainfall vs Yield"),use_container_width=True)
        st.dataframe(df.head(20),use_container_width=True)
with tab2:
    st.subheader("Enter growing conditions")
    c1,c2,c3=st.columns(3)
    with c1:
        crop=st.selectbox("Crop",["Rice","Wheat","Maize","Cotton","Sugarcane","Barley"])
        region=st.selectbox("Region",["Coastal","Northern","Central","Southern","Western"])
        season=st.selectbox("Season",["Kharif","Rabi","Summer"])
        soil=st.selectbox("Soil type",["Loamy","Clay","Sandy","Black","Alluvial"])
        irrigation=st.selectbox("Irrigation",["Rainfed","Canal","Drip","Borewell"])
    with c2:
        rain=st.slider("Rainfall (mm)",100,1800,700)
        temp=st.slider("Temperature (°C)",10,45,27)
        humidity=st.slider("Humidity (%)",20,100,65)
        area=st.number_input("Area (ha)",min_value=.2,max_value=100.0,value=1.0)
    with c3:
        nitrogen=st.slider("Nitrogen (kg/ha)",10,180,85)
        phosphorus=st.slider("Phosphorus (kg/ha)",5,100,42)
        potassium=st.slider("Potassium (kg/ha)",5,120,45)
        fertilizer=st.slider("Fertilizer (kg/ha)",10,220,105)
    if st.button("Predict Yield",type="primary"):
        row=pd.DataFrame([{"Crop":crop,"Region":region,"Season":season,"Soil_Type":soil,"Irrigation":irrigation,"Rainfall_mm":rain,"Temperature_C":temp,"Humidity_pct":humidity,"Nitrogen_kg_ha":nitrogen,"Phosphorus_kg_ha":phosphorus,"Potassium_kg_ha":potassium,"Fertilizer_kg_ha":fertilizer,"Area_ha":area}])
        val=float(model.predict(row)[0]); st.success(f"Estimated yield: {val:.2f} tons per hectare")
        st.caption("Prediction is an estimate from the included demonstration data, not agronomic advice.")
with tab3:
    st.subheader("Five-model cross-validation comparison")
    path=Path("reports/model_comparison.csv")
    if path.exists():
        results=pd.read_csv(path); st.dataframe(results,use_container_width=True)
        st.plotly_chart(px.bar(results,x="model",y="CV_R2",title="5-fold CV R² by Model"),use_container_width=True)
    else: st.info("Run train.py to create the model comparison report.")
