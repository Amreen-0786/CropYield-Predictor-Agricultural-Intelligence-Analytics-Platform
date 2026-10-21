import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

NUM=["Rainfall_mm","Temperature_C","Humidity_pct","Nitrogen_kg_ha","Phosphorus_kg_ha","Potassium_kg_ha","Fertilizer_kg_ha","Area_ha","NPK_Total","N_to_P_Ratio","K_to_N_Ratio","Humidity_Temperature_Index","Growing_Degree_Days"]
CAT=["Crop","Region","Season","Soil_Type","Irrigation","Rainfall_Category"]

class AgriFeatureEngineer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self,X):
        X=X.copy()
        X["NPK_Total"]=X[["Nitrogen_kg_ha","Phosphorus_kg_ha","Potassium_kg_ha"]].sum(axis=1,min_count=1)
        X["N_to_P_Ratio"]=X["Nitrogen_kg_ha"]/(X["Phosphorus_kg_ha"]+1)
        X["K_to_N_Ratio"]=X["Potassium_kg_ha"]/(X["Nitrogen_kg_ha"]+1)
        X["Rainfall_Category"]=pd.cut(X["Rainfall_mm"],[-np.inf,400,700,1000,np.inf],labels=["Low","Moderate","High","Very High"]).astype(object)
        X["Humidity_Temperature_Index"]=X["Humidity_pct"]*X["Temperature_C"]/100
        X["Growing_Degree_Days"]=np.maximum(X["Temperature_C"]-10,0)*120
        return X

def make_pipeline(model):
    prep=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),NUM),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),CAT)])
    return Pipeline([("features",AgriFeatureEngineer()),("preprocessor",prep),("model",model)])
