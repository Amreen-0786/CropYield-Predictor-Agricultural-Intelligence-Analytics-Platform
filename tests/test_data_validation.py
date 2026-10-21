from pathlib import Path
import pandas as pd
from src.cropyield_pipeline import AgriFeatureEngineer

def test_demo_dataset_schema_and_target():
    df=pd.read_csv("data/sample/crop_yield_demo.csv")
    assert "Yield_ton_ha" in df.columns
    assert len(df)>100
    assert df["Yield_ton_ha"].notna().all()

def test_feature_engineering_columns():
    df=pd.read_csv("data/sample/crop_yield_demo.csv").drop(columns="Yield_ton_ha").head(10)
    out=AgriFeatureEngineer().fit_transform(df)
    for col in ["NPK_Total","N_to_P_Ratio","K_to_N_Ratio","Rainfall_Category","Humidity_Temperature_Index","Growing_Degree_Days"]:
        assert col in out.columns

def test_yield_range():
    df=pd.read_csv("data/sample/crop_yield_demo.csv")
    assert (df["Yield_ton_ha"]>0).all()
