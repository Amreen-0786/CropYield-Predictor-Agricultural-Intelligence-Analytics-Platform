import json
from pathlib import Path
import joblib, pandas as pd
from sklearn.model_selection import KFold, cross_validate, train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.cropyield_pipeline import make_pipeline

def main():
    path=Path("data/raw/crop_yield.csv")
    if not path.exists(): path=Path("data/sample/crop_yield_demo.csv")
    df=pd.read_csv(path).drop_duplicates()
    if "Yield_ton_ha" not in df: raise ValueError("Dataset must contain Yield_ton_ha target.")
    df["Yield_ton_ha"]=pd.to_numeric(df["Yield_ton_ha"],errors="coerce")
    df=df.dropna(subset=["Yield_ton_ha"])
    X=df.drop(columns="Yield_ton_ha"); y=df["Yield_ton_ha"]
    models={"Linear Regression":LinearRegression(),"Ridge":Ridge(),
    "Random Forest":RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1),
    "Gradient Boosting":GradientBoostingRegressor(random_state=42),
    "Extra Trees":ExtraTreesRegressor(n_estimators=100,random_state=42,n_jobs=-1)}
    cv=KFold(n_splits=5,shuffle=True,random_state=42); rows=[]
    for name,est in models.items():
        scores=cross_validate(make_pipeline(est),X,y,cv=cv,scoring={"mae":"neg_mean_absolute_error","mse":"neg_mean_squared_error","r2":"r2"},n_jobs=1)
        rows.append({"model":name,"CV_MAE":-scores["test_mae"].mean(),"CV_RMSE":(-scores["test_mse"].mean())**.5,"CV_R2":scores["test_r2"].mean()})
    table=pd.DataFrame(rows).sort_values("CV_R2",ascending=False)
    best_name=table.iloc[0]["model"]; best_est=models[best_name]
    # Tune the best estimator if it is a tree model; pipeline keeps all transforms in CV folds.
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    base=make_pipeline(best_est)
    if best_name=="Random Forest":
        search=RandomizedSearchCV(base,{"model__n_estimators":[80,120,180],"model__max_depth":[None,10,16],"model__min_samples_leaf":[1,2,4]},n_iter=6,cv=3,scoring="r2",random_state=42,n_jobs=1)
        search.fit(Xtr,ytr); best=search.best_estimator_
        Path("reports/tuning_results.json").write_text(json.dumps({"best_params":search.best_params_,"best_cv_r2":search.best_score_},indent=2))
    else:
        best=base.fit(Xtr,ytr)
        Path("reports/tuning_results.json").write_text(json.dumps({"note":"Best CV model was not Random Forest; default model selected. Use RandomizedSearchCV on a tree model if desired."},indent=2))
    best.fit(Xtr,ytr); pred=best.predict(Xte)
    metrics={"best_model":best_name,"MAE":mean_absolute_error(yte,pred),"RMSE":mean_squared_error(yte,pred)**.5,"R2":r2_score(yte,pred)}
    Path("models").mkdir(exist_ok=True); joblib.dump(best,"models/cropyield_pipeline.joblib")
    Path("reports").mkdir(exist_ok=True); table.to_csv("reports/model_comparison.csv",index=False)
    Path("reports/holdout_metrics.json").write_text(json.dumps(metrics,indent=2))
    print(table.to_string(index=False)); print(metrics)
if __name__=="__main__": main()
