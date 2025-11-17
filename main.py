import os
from io import BytesIO
from typing import List, Optional, Dict, Any

# Light imports required to boot
import chardet
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Optional heavy scientific stack (lazy/guarded)
try:
    from sklearn.model_selection import train_test_split, KFold, cross_val_score  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # type: ignore
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet  # type: ignore
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
    LinearRegression = Ridge = Lasso = ElasticNet = RandomForestRegressor = GradientBoostingRegressor = None  # type: ignore

# Optional xgboost
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    XGBRegressor = None  # type: ignore
    HAS_XGB = False

# Optional statsmodels + scipy
try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white  # type: ignore
    from statsmodels.stats.stattools import durbin_watson  # type: ignore
    from scipy import stats  # type: ignore
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False
    sm = variance_inflation_factor = het_breuschpagan = het_white = durbin_watson = stats = None  # type: ignore

# Optional plotting stack
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False
    plt = sns = None  # type: ignore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for uploaded dataset per session key (demo only)
DATA_STORE: Dict[str, pd.DataFrame] = {}

class UploadSummary(BaseModel):
    n_rows: int
    n_cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_per_column: Dict[str, int]
    desc_stats: Dict[str, Dict[str, float]]


def detect_encoding_and_delimiter(content: bytes) -> Dict[str, Optional[str]]:
    result = chardet.detect(content)
    encoding = result.get('encoding') or 'utf-8'
    sample = content[:4000].decode(encoding, errors='ignore')
    lines = sample.splitlines()
    header = lines[0] if lines else ''
    candidates = [',', ';', '\t', '|']
    best_delim = max(candidates, key=lambda d: header.count(d)) if header else ','
    return {"encoding": encoding, "delimiter": best_delim}


def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    filename = (file.filename or "").lower()
    content = file.file.read()
    if filename.endswith('.xlsx'):
        return pd.read_excel(BytesIO(content))
    info = detect_encoding_and_delimiter(content)
    encoding = info["encoding"] or 'utf-8'
    delim = info["delimiter"] or ','
    return pd.read_csv(BytesIO(content), encoding=encoding, sep=delim)


def summarize_dataframe(df: pd.DataFrame) -> UploadSummary:
    dtypes = {c: ("datetime" if np.issubdtype(dt, np.datetime64) else str(dt)) for c, dt in df.dtypes.items()}
    missing = df.isna().sum().to_dict()
    desc = df.select_dtypes(include=[np.number]).describe().to_dict()
    return UploadSummary(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=list(df.columns),
        dtypes=dtypes,
        missing_per_column=missing,
        desc_stats=desc
    )


@app.get("/")
def read_root():
    return {
        "message": "Regression backend ready",
        "modules": {
            "sklearn": HAS_SKLEARN,
            "xgboost": HAS_XGB,
            "statsmodels": HAS_STATSMODELS,
            "plotting": HAS_PLOTTING
        }
    }


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...), session_key: str = Form("default")):
    try:
        df = read_uploaded_file(file)
        # Try to parse dates from object columns
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    parsed = pd.to_datetime(df[col], errors='raise', infer_datetime_format=True)
                    if parsed.notna().mean() > 0.8:
                        df[col] = parsed
                except Exception:
                    pass
        DATA_STORE[session_key] = df
        summary = summarize_dataframe(df)
        preview = df.head(100).fillna("").to_dict(orient='records')
        return {"summary": summary.model_dump(), "preview": preview}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


class TransformRequest(BaseModel):
    session_key: str = "default"
    target: str
    predictors: List[str]
    transforms: Dict[str, str] = {}
    standardize: bool = False
    dummy_encode: bool = True
    imputation: Optional[str] = None  # mean, median, most_frequent


def apply_transforms(df: pd.DataFrame, req: TransformRequest):
    df = df.copy()
    for col, t in (req.transforms or {}).items():
        if col in df.columns:
            if t == 'log':
                df[col] = np.log1p(pd.to_numeric(df[col], errors='coerce').clip(lower=0))
            elif t == 'sqrt':
                df[col] = np.sqrt(pd.to_numeric(df[col], errors='coerce').clip(lower=0))
    return df


class ModelRequest(BaseModel):
    session_key: str = "default"
    target: str
    predictors: List[str]
    transforms: Dict[str, str] = {}
    standardize: bool = False
    dummy_encode: bool = True
    imputation: Optional[str] = None
    test_size: float = 0.2
    cv_folds: int = 5
    models: List[str] = ["ols", "ridge", "lasso", "elasticnet", "rf", "gbr", "xgb"]
    selection_metric: str = "rmse"  # rmse, mae, r2


def ensure_modules(require_stats: bool = False, require_plots: bool = False):
    if not HAS_SKLEARN:
        return JSONResponse(status_code=500, content={"error": "scikit-learn não está disponível no servidor."})
    if require_stats and not HAS_STATSMODELS:
        return JSONResponse(status_code=500, content={"error": "statsmodels/scipy não está disponível no servidor."})
    if require_plots and not HAS_PLOTTING:
        return JSONResponse(status_code=500, content={"error": "Stack de plots (matplotlib/seaborn) não está disponível."})
    return None


def compute_vif(X: pd.DataFrame) -> Dict[str, float]:
    if not HAS_STATSMODELS:
        return {}
    vifs = {}
    X = X.dropna()
    if X.shape[1] <= 1:
        return vifs
    X_const = sm.add_constant(X)
    for i, col in enumerate(X.columns):
        try:
            vifs[col] = float(variance_inflation_factor(X_const.values, i + 1))
        except Exception:
            vifs[col] = float('nan')
    return vifs


def run_linear_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, X: pd.DataFrame) -> Dict[str, Any]:
    if not HAS_STATSMODELS:
        return {}
    resid = y_true - y_pred
    out: Dict[str, Any] = {}
    try:
        shapiro_stat, shapiro_p = stats.shapiro(resid)
        out['shapiro'] = {"stat": float(shapiro_stat), "p": float(shapiro_p)}
    except Exception:
        out['shapiro'] = {"stat": None, "p": None}
    try:
        from statsmodels.stats.diagnostic import normal_ad  # type: ignore
        ad_stat, ad_p = normal_ad(resid)
        out['anderson_darling'] = {"stat": float(ad_stat), "p": float(ad_p)}
    except Exception:
        out['anderson_darling'] = {"stat": None, "p": None}
    try:
        lm = sm.OLS(y_true, sm.add_constant(X)).fit()
        bp = het_breuschpagan(lm.resid, lm.model.exog)
        out['breusch_pagan'] = {"lm_stat": float(bp[0]), "lm_p": float(bp[1]), "f_stat": float(bp[2]), "f_p": float(bp[3])}
        wt = het_white(lm.resid, lm.model.exog)
        out['white'] = {"stat": float(wt[0]), "p": float(wt[1])}
        out['f_test'] = {"fvalue": float(lm.fvalue) if lm.fvalue is not None else None, "fpvalue": float(lm.f_pvalue) if lm.f_pvalue is not None else None}
        out['t_tests'] = {name: {"coef": float(coef), "t": float(t), "p": float(p)} for name, coef, t, p in zip(lm.params.index, lm.params.values, lm.tvalues.values, lm.pvalues.values)}
        out['conf_int'] = {name: [float(ci[0]), float(ci[1])] for name, ci in lm.conf_int().iterrows()}
        out['durbin_watson'] = float(durbin_watson(lm.resid))
    except Exception:
        out['breusch_pagan'] = out['white'] = out['f_test'] = out['t_tests'] = out['conf_int'] = None
    try:
        vifs = compute_vif(X.select_dtypes(include=[np.number]))
        out['vif'] = vifs
    except Exception:
        out['vif'] = None
    return out


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test, cv_folds: int, selection_metric: str) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    p = X_test.shape[1]
    n = X_test.shape[0]
    adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else float('nan')

    try:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring_map = {
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        scoring = scoring_map.get(selection_metric, 'neg_root_mean_squared_error')
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_metric = float(np.mean(-cv_scores)) if selection_metric in ['rmse', 'mae'] else float(np.mean(cv_scores))
    except Exception:
        cv_metric = None

    return {
        'name': name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adj_r2': adj_r2,
        'cv_metric': cv_metric,
        'n_test': int(n)
    }


@app.post("/api/model")
async def model_pipeline(req: ModelRequest):
    err = ensure_modules(require_stats=True)
    if err:
        return err
    if req.session_key not in DATA_STORE:
        return JSONResponse(status_code=400, content={"error": "No dataset uploaded for this session_key"})

    df = DATA_STORE[req.session_key].copy()
    if req.target not in df.columns:
        return JSONResponse(status_code=400, content={"error": "Target not in dataset"})

    # Apply transforms
    df = apply_transforms(df, TransformRequest(**req.model_dump()))

    features = [c for c in req.predictors if c in df.columns]
    if not features:
        return JSONResponse(status_code=400, content={"error": "Nenhum preditor válido informado"})
    df = df[features + [req.target]].copy()

    y = pd.to_numeric(df[req.target], errors='coerce')
    X = df.drop(columns=[req.target])
    X = pd.get_dummies(X, drop_first=True) if req.dummy_encode else X

    # Imputation
    if req.imputation in ['mean', 'median', 'most_frequent']:
        strategy = req.imputation
        imp = SimpleImputer(strategy=strategy)
        X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
        y = pd.Series(SimpleImputer(strategy='median').fit_transform(y.to_frame()).ravel())
    else:
        X = X.fillna(X.median(numeric_only=True))
        y = y.fillna(y.median())

    if req.standardize:
        scaler = StandardScaler()
        X[X.columns] = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size, random_state=42)

    results: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Any] = {}

    models_map: Dict[str, Any] = {
        'ols': LinearRegression(),
        'ridge': Ridge(alpha=1.0, random_state=42),
        'lasso': Lasso(alpha=0.01, random_state=42, max_iter=10000),
        'elasticnet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000),
        'rf': RandomForestRegressor(n_estimators=300, random_state=42),
        'gbr': GradientBoostingRegressor(random_state=42),
    }
    if HAS_XGB and 'xgb' in req.models:
        models_map['xgb'] = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2)

    for name in req.models:
        if name not in models_map:
            continue
        model = models_map[name]
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test, req.cv_folds, req.selection_metric)
        results.append(res)
        try:
            model.fit(X, y)
            fitted_models[name] = model
        except Exception:
            pass

    # Select best model
    if req.selection_metric in ['rmse', 'mae']:
        valid = [r for r in results if r['cv_metric'] is not None]
        best = min(valid, key=lambda x: x['cv_metric']) if valid else min(results, key=lambda x: x['rmse'])
    elif req.selection_metric in ['r2']:
        valid = [r for r in results if r['cv_metric'] is not None]
        best = max(valid, key=lambda x: x['cv_metric']) if valid else max(results, key=lambda x: x['r2'])
    else:
        best = min(results, key=lambda x: x['rmse'])

    explanation = f"Modelo {best['name']} selecionado pela métrica {req.selection_metric.upper()} com valor {best['cv_metric'] if best['cv_metric'] is not None else best[req.selection_metric]:.4f}."

    diagnostics = None
    if 'ols' in fitted_models and HAS_STATSMODELS:
        ols_model = fitted_models['ols']
        y_hat = ols_model.predict(X)
        diagnostics = run_linear_diagnostics(y.values, y_hat, X)

    return {"results": results, "best": best, "explanation": explanation, "diagnostics": diagnostics}


@app.post('/api/correlogram')
async def correlogram(session_key: str = Form("default")):
    err = ensure_modules(require_plots=True)
    if err:
        return err
    if session_key not in DATA_STORE:
        return JSONResponse(status_code=400, content={"error": "No dataset uploaded"})
    df = DATA_STORE[session_key].copy()
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.empty:
        return JSONResponse(status_code=400, content={"error": "Não há colunas numéricas suficientes"})
    corr = num_df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, square=True)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=160)
    plt.close()
    buf.seek(0)
    return JSONResponse(content={"image": buf.getvalue().hex()})


@app.post('/api/diagnostic-plots')
async def diagnostic_plots(session_key: str = Form("default"), target: str = Form(...), predictors: str = Form(...)):
    err = ensure_modules(require_stats=True, require_plots=True)
    if err:
        return err
    if session_key not in DATA_STORE:
        return JSONResponse(status_code=400, content={"error": "No dataset uploaded"})
    df = DATA_STORE[session_key].copy()
    predictors_list = [p.strip() for p in predictors.split(',') if p.strip() in df.columns]
    if target not in df.columns or not predictors_list:
        return JSONResponse(status_code=400, content={"error": "Target ou preditores inválidos"})
    X = pd.get_dummies(df[predictors_list], drop_first=True)
    y = pd.to_numeric(df[target], errors='coerce')
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.median())
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    resid = y - y_pred

    plots = {}

    # Residuals vs Fitted
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=y_pred, y=resid, s=20)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Ajustado')
    plt.ylabel('Resíduo')
    buf = BytesIO()
    plt.tight_layout(); plt.savefig(buf, format='png', dpi=160); plt.close(); buf.seek(0)
    plots['resid_vs_fitted'] = buf.getvalue().hex()

    # QQ plot
    plt.figure(figsize=(6,4))
    sm.qqplot(resid, line='45', fit=True)
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format='png', dpi=160); plt.close(); buf.seek(0)
    plots['qqplot'] = buf.getvalue().hex()

    # Cook's distance / leverage plot
    lm = sm.OLS(y, sm.add_constant(X)).fit()
    infl = lm.get_influence()
    cooks = infl.cooks_distance[0]
    leverage = infl.hat_matrix_diag
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=leverage, y=cooks, s=20)
    plt.xlabel('Leverage'); plt.ylabel("Cook's distance")
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png', dpi=160); plt.close(); buf.seek(0)
    plots['cooks_leverage'] = buf.getvalue().hex()

    return {"plots": plots}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
