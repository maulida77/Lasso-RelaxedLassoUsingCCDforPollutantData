import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import os, time
begin = time.time()

# ============================================================
# SETTINGS
# ============================================================
path_data = 'Result/pollutan/'
os.makedirs(path_data, exist_ok=True)
os.makedirs(path_data + 'sensitivity_tuning/', exist_ok=True)

features = ['NOx','SO₂','CO','OC','NMVOC','BC','NH₃']
output = 'Y'
INITIAL_TRAIN_END_YEAR = 2005
FINAL_RULE = '1SE'                 # '1SE' or 'MIN'
tol = 1e-6
gamma_grid = np.linspace(0, 1, 11)
n_lambda = 100
lambda_min_ratio = 1e-3
splits = [
    {'id':'S1','train_end':2014,'test_start':2015,'label':'1990-2014 / 2015-2021'},
    {'id':'S2','train_end':2015,'test_start':2016,'label':'1990-2015 / 2016-2021'},
    {'id':'S3','train_end':2016,'test_start':2017,'label':'1990-2016 / 2017-2021'}]
lags = [0, 1, 2]

# ============================================================
# BASIC FUNCTIONS: CUSTOM CYCLIC COORDINATE DESCENT
# ============================================================
def soft_threshold(rho, lamda):
    if rho < -lamda: return rho + lamda
    if rho > lamda: return rho - lamda
    return 0.0

def standardize_train(X):
    mean = X.mean(axis=0); std = X.std(axis=0, ddof=0)#; std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std

def coordinate_descent_lasso(theta, X, y, lamda, tol=1e-6, max_iter=100000):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1).copy(); n = X.shape[1]
    residual = y - X @ theta
    for _ in range(max_iter):
        max_change = 0.0
        for j in range(n):
            old = theta[j]; residual += X[:, j] * old
            rho = np.mean(X[:, j] * residual)
            new = soft_threshold(rho, lamda)
            theta[j] = new; residual -= X[:, j] * new
            max_change = max(max_change, abs(new - old))
        if max_change < tol: break
    return theta

def theta_LS_full(theta_lasso, X_standardized, y_centered):
    theta_ls = np.zeros(X_standardized.shape[1]); active = np.where(np.abs(theta_lasso) > 1e-10)[0] 
    if len(active) > 0:
        fit = linear_model.LinearRegression(fit_intercept=False).fit(X_standardized[:, active], y_centered)
        theta_ls[active] = fit.coef_
    return theta_ls

def lasso_components(X_train, y_train, lamda, tol=1e-6):
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    X_std, X_mean, X_sd = standardize_train(X_train); y_mean = y_train.mean(); y_centered = y_train - y_mean
    beta_lasso_std = coordinate_descent_lasso(np.zeros(X_train.shape[1]), X_std, y_centered, lamda, tol)
    beta_ls_std = theta_LS_full(beta_lasso_std, X_std, y_centered)
    return {'beta_lasso_std':beta_lasso_std,'beta_ls_std':beta_ls_std,'X_mean':X_mean,'X_sd':X_sd,'y_mean':y_mean}

def model_from_components(comp, gamma=1.0):
    beta_std = gamma * comp['beta_lasso_std'] + (1 - gamma) * comp['beta_ls_std']
    beta = beta_std / comp['X_sd']; intercept = comp['y_mean'] - comp['X_mean'] @ beta
    return {'beta_std':beta_std,'beta':beta,'intercept':intercept,'nnz':int(np.sum(np.abs(beta) > 1e-10))} 

def fit_lasso(X_train, y_train, lamda, tol=1e-6):
    return model_from_components(lasso_components(X_train, y_train, lamda, tol), gamma=1.0)

def fit_relaxed_lasso(X_train, y_train, lamda, gamma, tol=1e-6):
    return model_from_components(lasso_components(X_train, y_train, lamda, tol), gamma=gamma)

def fit_ols(X_train, y_train):
    # OLS is the unpenalized (lambda=0) benchmark; solve directly for numerical stability.
    y_train=np.asarray(y_train,dtype=float).reshape(-1); X_std,X_mean,X_sd=standardize_train(X_train)
    y_mean=y_train.mean(); beta_std=np.linalg.lstsq(X_std,y_train-y_mean,rcond=None)[0]
    beta=beta_std/X_sd; intercept=y_mean-X_mean@beta
    return {'beta_std':beta_std,'beta':beta,'intercept':intercept,'nnz':X_train.shape[1]}

def predict_output(X, beta, intercept): return intercept + X @ beta

# ============================================================
# METRICS
# ============================================================
def MSE(y, pred):
    y = np.asarray(y).reshape(-1); pred = np.asarray(pred).reshape(-1); return np.mean((y - pred)**2)

def RMSE(y, pred): return np.sqrt(MSE(y, pred))

def MAE(y, pred):
    y = np.asarray(y).reshape(-1); pred = np.asarray(pred).reshape(-1); return np.mean(np.abs(y - pred))

def calculate_lambda_max(X_train, y_train):
    X_std, _, _ = standardize_train(X_train); yc = np.asarray(y_train).reshape(-1) - np.mean(y_train)
    return np.max(np.abs(X_std.T @ yc)) / len(y_train)

def selected_feature_names(beta, names, threshold=1e-10): 
    idx = np.where(np.abs(beta) > threshold)[0]; return ', '.join(names[i] for i in idx) if len(idx) else 'None'

# ============================================================
# EXPANDING-WINDOW TEMPORAL CV
# ============================================================
def temporal_cv_lasso(X, y, years, lamda, initial_train_end=2005, tol=1e-6):
    errors, predictions = [], []
    for valid_year in years[(years > initial_train_end) & (years <= years.max())]:
        tr, va = years < valid_year, years == valid_year
        model = fit_lasso(X[tr], y[tr], lamda, tol); pred = predict_output(X[va], model['beta'], model['intercept'])
        err = MSE(y[va], pred); errors.append(err)
        predictions.append({'Year':int(valid_year),'Observed':float(y[va][0]),'Predicted':float(pred[0]),'Squared_Error':float(err)})
    return np.mean(errors), np.asarray(errors), pd.DataFrame(predictions)

def temporal_cv_relaxed(X, y, years, lamda, gamma, initial_train_end=2005, tol=1e-6):
    errors, predictions = [], []
    for valid_year in years[(years > initial_train_end) & (years <= years.max())]:
        tr, va = years < valid_year, years == valid_year
        model = fit_relaxed_lasso(X[tr], y[tr], lamda, gamma, tol); pred = predict_output(X[va], model['beta'], model['intercept'])
        err = MSE(y[va], pred); errors.append(err)
        predictions.append({'Year':int(valid_year),'Observed':float(y[va][0]),'Predicted':float(pred[0]),'Squared_Error':float(err)})
    return np.mean(errors), np.asarray(errors), pd.DataFrame(predictions)

def temporal_cv_ols(X, y, years, initial_train_end=2005):
    predictions=[]
    for valid_year in years[(years>initial_train_end)&(years<=years.max())]:
        tr,va=years<valid_year,years==valid_year; model=fit_ols(X[tr],y[tr]); pred=predict_output(X[va],model['beta'],model['intercept'])
        predictions.append({'Year':int(valid_year),'Observed':float(y[va][0]),'Predicted':float(pred[0])})
    return pd.DataFrame(predictions)

def cv_metrics(pred_df):
    obs=pred_df['Observed'].to_numpy(); pred=pred_df['Predicted'].to_numpy()
    return RMSE(obs,pred),MAE(obs,pred)

# ============================================================
# HYPERPARAMETER TUNING: CUSTOM CCD WITH PATHWISE WARM STARTS
# ============================================================
def select_min_and_1se(results):
    best = results.loc[results['CV_MSE'].idxmin()]
    best_by_nnz = results.loc[results.groupby('NNZ')['CV_MSE'].idxmin()].sort_values('NNZ').reset_index(drop=True)
    threshold = best['CV_MSE'] + best['Fold_SE_MSE']; eligible = results[results['CV_MSE'] <= threshold]
    smallest_nnz = eligible['NNZ'].min(); candidates = eligible[eligible['NNZ'] == smallest_nnz]
    one_se = candidates.loc[candidates['CV_MSE'].idxmin()]
    return best, best_by_nnz, one_se

def lasso_component_path(X_train, y_train, lambda_grid, tol=1e-6):
    """Custom cyclic coordinate-descent path; previous lambda solution is used as the warm start."""
    y_train = np.asarray(y_train,dtype=float).reshape(-1); X_std,X_mean,X_sd=standardize_train(X_train)
    y_mean=y_train.mean(); yc=y_train-y_mean; theta=np.zeros(X_train.shape[1]); path=[]
    for lamda in lambda_grid:  # lambda_grid is descending from lambda_max
        theta=coordinate_descent_lasso(theta,X_std,yc,lamda,tol)
        path.append({'beta_lasso_std':theta.copy(),'beta_ls_std':theta_LS_full(theta,X_std,yc),'X_mean':X_mean,'X_sd':X_sd,'y_mean':y_mean})
    return path

def tune_models(X_train,y_train,years_train,lambda_grid,gamma_grid,initial_train_end=2005,tol=1e-6):
    """Tune LASSO and relaxed LASSO together without changing the custom CCD estimator."""
    full_path=lasso_component_path(X_train,y_train,lambda_grid,tol)
    nlam=len(lambda_grid)
    ngam=len(gamma_grid)
    lasso_err=[[] for _ in range(nlam)]
    relax_err=[[[] for _ in range(ngam)] for _ in range(nlam)]
    validation_years=years_train[(years_train>initial_train_end)&(years_train<=years_train.max())]
    for valid_year in validation_years:
        tr,va=years_train<valid_year,years_train==valid_year
        fold_path=lasso_component_path(X_train[tr],y_train[tr],lambda_grid,tol)
        for i,comp in enumerate(fold_path):
            ml=model_from_components(comp,1.0)
            pl=predict_output(X_train[va],ml['beta'],ml['intercept'])
            lasso_err[i].append(MSE(y_train[va],pl))
            for k,gamma in enumerate(gamma_grid):
                mr=model_from_components(comp,float(gamma))
                pr=predict_output(X_train[va],mr['beta'],mr['intercept'])
                relax_err[i][k].append(MSE(y_train[va],pr))
    lasso_rows=[]; relax_rows=[]
    for i,lamda in enumerate(lambda_grid):
        lasso_model = model_from_components(full_path[i], 1.0)
        nnz_lasso = lasso_model['nnz']
        e=np.asarray(lasso_err[i]); mm=e.mean()
        lasso_rows.append({'lambda':lamda,'NNZ':nnz_lasso,'CV_MSE':mm,'CV_RMSE':np.sqrt(mm),'Fold_SD_MSE':np.std(e,ddof=1),'Fold_SE_MSE':np.std(e,ddof=1)/np.sqrt(len(e))})
        for k,gamma in enumerate(gamma_grid):
            relax_model = model_from_components(full_path[i], float(gamma))
            nnz_relax = relax_model['nnz']
            er=np.asarray(relax_err[i][k])
            mmr=er.mean()
            relax_rows.append({'lambda':lamda,'gamma':float(gamma),'NNZ':nnz_relax,'CV_MSE':mmr,'CV_RMSE':np.sqrt(mmr),'Fold_SD_MSE':np.std(er,ddof=1),'Fold_SE_MSE':np.std(er,ddof=1)/np.sqrt(len(er))})
    lasso_results=pd.DataFrame(lasso_rows); relax_results=pd.DataFrame(relax_rows)
    lb,lbn,l1=select_min_and_1se(lasso_results); rb,rbn,r1=select_min_and_1se(relax_results)
    return (lasso_results,lb,lbn,l1),(relax_results,rb,rbn,r1),full_path

# ============================================================
# DIAGNOSTICS
# ============================================================
def calculate_vif(X, feature_names):
    rows = []
    for j in range(X.shape[1]):
        response = X[:, j]; predictors = np.delete(X, j, axis=1)
        r2 = linear_model.LinearRegression().fit(predictors, response).score(predictors, response)
        rows.append({'Predictor':feature_names[j],'VIF':np.inf if (1-r2)<=1e-12 else 1/(1-r2)})
    return pd.DataFrame(rows)

#add
def lag1_autocorrelation(residuals):
    residuals = np.asarray(residuals,dtype=float).reshape(-1)
    if len(residuals) < 3 or np.std(residuals[:-1]) == 0 or np.std(residuals[1:]) == 0:
        return np.nan
    return np.corrcoef(residuals[:-1],residuals[1:])[0,1]

def residual_diagnostics(pred_df):
    obs = pred_df['Observed'].to_numpy(dtype=float); pred = pred_df['Predicted'].to_numpy(dtype=float)
    resid = obs - pred
    return {'N':len(resid),'Residual_Mean':np.mean(resid),'Residual_SD':np.std(resid,ddof=1),'Lag1_Residual_ACF':lag1_autocorrelation(resid)}

def linear_quadratic_assessment(x, y):
    x = np.asarray(x,dtype=float).reshape(-1); y = np.asarray(y,dtype=float).reshape(-1)
    Xlin = x.reshape(-1,1)
    lin = linear_model.LinearRegression().fit(Xlin,y); pred_lin = lin.predict(Xlin)
    Xquad = np.column_stack((x,x**2))
    quad = linear_model.LinearRegression().fit(Xquad,y); pred_quad = quad.predict(Xquad)
    sst = np.sum((y-y.mean())**2)
    r2_lin = np.nan if sst == 0 else 1 - np.sum((y-pred_lin)**2)/sst
    r2_quad = np.nan if sst == 0 else 1 - np.sum((y-pred_quad)**2)/sst
    return r2_lin,r2_quad,r2_quad-r2_lin

# ============================================================
# LAGGED DATA + ONE COMPLETE TEMPORAL SPECIFICATION
# ============================================================
def create_lagged_dataset(X_data, y_data, years_data, lag):
    if lag == 0: return X_data.copy(), y_data.copy(), years_data.copy()
    return X_data[:-lag].copy(), y_data[lag:].copy(), years_data[lag:].copy()

def run_specification(X_data, y_data, years_data, split, lag, feature_names, final_rule='1SE'):
    X_lag,y_lag,years_lag=create_lagged_dataset(X_data,y_data,years_data,lag); tr=years_lag<=split['train_end']; te=years_lag>=split['test_start']
    X_tr,y_tr,years_tr=X_lag[tr],y_lag[tr],years_lag[tr]
    X_te,y_te,years_te=X_lag[te],y_lag[te],years_lag[te]
    lam_max=calculate_lambda_max(X_tr,y_tr) 
    lam_grid=np.logspace(np.log10(lam_max),np.log10(lam_max*lambda_min_ratio),n_lambda)
    lasso_tuned,relax_tuned,full_path=tune_models(X_tr,y_tr,years_tr,lam_grid,gamma_grid,INITIAL_TRAIN_END_YEAR,tol)
    lasso_all,lasso_min,lasso_by_nnz,lasso_1se=lasso_tuned
    relax_all,relax_min,relax_by_nnz,relax_1se=relax_tuned
    lc=lasso_1se if final_rule=='1SE' else lasso_min
    rc=relax_1se if final_rule=='1SE' else relax_min
    lam_lasso=float(lc['lambda'])
    lam_relax=float(rc['lambda'])
    gamma_relax=float(rc['gamma'])
    ols=fit_ols(X_tr,y_tr) 
    lasso=fit_lasso(X_tr,y_tr,lam_lasso,tol)
    relax=fit_relaxed_lasso(X_tr,y_tr,lam_relax,gamma_relax,tol)
    train_pred={'OLS':predict_output(X_tr,ols['beta'],ols['intercept']),'LASSO':predict_output(X_tr,lasso['beta'],lasso['intercept']),'Relaxed LASSO':predict_output(X_tr,relax['beta'],relax['intercept'])}
    test_pred={'OLS':predict_output(X_te,ols['beta'],ols['intercept']),'LASSO':predict_output(X_te,lasso['beta'],lasso['intercept']),'Relaxed LASSO':predict_output(X_te,relax['beta'],relax['intercept'])}
    cvpred_ols=temporal_cv_ols(X_tr,y_tr,years_tr,INITIAL_TRAIN_END_YEAR)
    _,_,cvpred_lasso=temporal_cv_lasso(X_tr,y_tr,years_tr,lam_lasso,INITIAL_TRAIN_END_YEAR,tol)
    _,_,cvpred_relax=temporal_cv_relaxed(X_tr,y_tr,years_tr,lam_relax,gamma_relax,INITIAL_TRAIN_END_YEAR,tol)
    cv={'OLS':cv_metrics(cvpred_ols),'LASSO':cv_metrics(cvpred_lasso),'Relaxed LASSO':cv_metrics(cvpred_relax)}
    case=f"{split['id']}L{lag}"; rows=[]
    for model,obj,lam,gam,selected in [('OLS',ols,np.nan,np.nan,'All'),('LASSO',lasso,lam_lasso,np.nan,selected_feature_names(lasso['beta'],feature_names)),('Relaxed LASSO',relax,lam_relax,gamma_relax,selected_feature_names(relax['beta'],feature_names))]:
        p=obj['nnz']
        rows.append({'Case':case,'Split':split['label'],'Lag':lag,'Model':model,'lambda':lam,'gamma':gam,'NNZ':p,'Selected_predictors':selected,
            'CV_RMSE':cv[model][0],'CV_MAE':cv[model][1],
            'Train_RMSE':RMSE(y_tr,train_pred[model]),'Train_MAE':MAE(y_tr,train_pred[model]),
            'Test_RMSE':RMSE(y_te,test_pred[model]),'Test_MAE':MAE(y_te,test_pred[model])})
    summary=pd.DataFrame(rows)
    predictions=pd.DataFrame({'Case':case,'Split':split['label'],'Lag':lag,'Year':years_te,'Observed':y_te,'OLS':test_pred['OLS'],'LASSO':test_pred['LASSO'],'Relaxed_LASSO':test_pred['Relaxed LASSO']})
    return {'case':case,'split':split,'lag':lag,'summary':summary,'predictions':predictions,'betas':{'LASSO':lasso['beta'].copy(),'Relaxed LASSO':relax['beta'].copy()},
        'models':{'OLS':ols,'LASSO':lasso,'Relaxed LASSO':relax},'lambda_grid':lam_grid,'full_path':full_path,
        'lasso_all':lasso_all,'lasso_min':lasso_min,'lasso_1se':lasso_1se,'lasso_by_nnz':lasso_by_nnz,
        'relax_all':relax_all,'relax_min':relax_min,'relax_1se':relax_1se,'relax_by_nnz':relax_by_nnz,
        'cvpred_ols':cvpred_ols,'cvpred_lasso':cvpred_lasso,'cvpred_relax':cvpred_relax}

# ============================================================
# LOAD DATA
# ============================================================
data = pd.read_csv('pollutan.csv')
if 'year' not in data.columns:
    if len(data) != 32: raise ValueError('Expected 32 observations for 1990-2021.')
    data.insert(0, 'year', np.arange(1990, 2022))
data = data.sort_values('year').reset_index(drop=True)
if data['year'].iloc[0] != 1990 or data['year'].iloc[-1] != 2021 or len(data) != 32: raise ValueError('Data must cover 1990-2021 with 32 annual observations.')
years = data['year'].to_numpy(); X_original = data[features].to_numpy(dtype=float); y = data[output].to_numpy(dtype=float)
X = X_original / 100000.0  # reporting scale: one predictor unit = 10^5 tonnes; standardized LASSO fit is unchanged

# ============================================================
# 4.1 DATA DIAGNOSTICS
# ============================================================
#TABLE 1
vif_table = calculate_vif(X_original, features); vif_table.to_csv(path_data+'Table1_VIF.csv',index=False)

#add
# SUPPLEMENTARY NONLINEARITY CHECK: marginal linear vs quadratic fits
nonlinearity_table = pd.DataFrame([
    {'Predictor':features[j],
     'Linear_R2':linear_quadratic_assessment(X_original[:,j],y)[0],
     'Quadratic_R2':linear_quadratic_assessment(X_original[:,j],y)[1],
     'Delta_R2':linear_quadratic_assessment(X_original[:,j],y)[2]}
    for j in range(len(features))
])
nonlinearity_table.to_csv(path_data+'Supplement_nonlinearity_diagnostic.csv',index=False)

#FIGURE 1
plt.figure(figsize=(5.5,3.3))
for j,name in enumerate(features):
    plt.plot(years,X_original[:,j],marker='o',markersize=2.5,linewidth=1.2,label=name)
plt.xlabel('Year',fontsize=9)
plt.ylabel('National annual emissions (tonnes)',fontsize=9)
plt.yscale('log')
plt.xticks(np.arange(1990,2022,5),fontsize=8)
plt.yticks(fontsize=8)
plt.legend(ncol=2,fontsize=7.5)
plt.tight_layout()
plt.savefig(path_data+'Fig1_national_emissions_by_year.png',dpi=300,bbox_inches='tight')
plt.close()

#FIGURE 2
fig,axes=plt.subplots(1,2,figsize=(6.5,3.4),sharey=True)
norm=plt.Normalize(years.min(),years.max())
cmap=plt.cm.Greens
sc1=axes[0].scatter(X_original[:,1]/1e5,y,c=years,cmap=cmap,norm=norm,s=26)
axes[0].set_xlabel(r'SO$_2$ emissions ($10^5$ tonnes)',fontsize=9)
sc2=axes[1].scatter(X_original[:,3]/1e5,y,c=years,cmap=cmap,norm=norm,s=26)
axes[1].set_xlabel(r'OC emissions ($10^5$ tonnes)',fontsize=9)
for ax in axes:
    ax.tick_params(axis='both',labelsize=8)
    ax.grid(alpha=.2)
fig.supylabel('Air-pollution-related death rate\n(per 100,000 population)',x=0.02,fontsize=9)
fig.subplots_adjust(left=.13,right=.86,bottom=.15,top=.96,wspace=.10)
cbar=fig.colorbar(sc2,ax=axes,pad=.035,fraction=.035)
cbar.set_label('Year',fontsize=9)
cbar.ax.tick_params(labelsize=8)
plt.savefig(path_data+'Fig2_death_rate_SO2_OC_by_year.png',dpi=300,bbox_inches='tight')
plt.close()

#FIGURE3
corr = pd.DataFrame(X_original,columns=features).corr(); fig,ax=plt.subplots(figsize=(6.5,5.4)); im=ax.imshow(corr.values,vmin=-1,vmax=1,cmap='coolwarm'); ax.set_xticks(range(len(features)));
ax.set_yticks(range(len(features))); ax.set_xticklabels(features,rotation=45,ha='right'); ax.set_yticklabels(features)
for i in range(len(features)):
    for j in range(len(features)): ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7)
fig.colorbar(im,ax=ax,label='Pearson correlation'); plt.tight_layout(); plt.savefig(path_data+'Fig3_predictor_correlation_heatmap.png',dpi=300,bbox_inches='tight'); plt.close()

# ============================================================
# 4.2-4.5 RUN ALL 3 SPLITS x 3 LAGS ONCE
# ============================================================
runs = {}; summaries = []; predictions_all = []
for split in splits:
    for lag in lags:
        run = run_specification(X,y,years,split,lag,features,FINAL_RULE); runs[(split['id'],lag)] = run; summaries.append(run['summary']); predictions_all.append(run['predictions'])
        safe = f"{split['id']}_lag{lag}"
        run['lasso_all'].to_csv(path_data+f'sensitivity_tuning/{safe}_lasso_all.csv',index=False)
        run['relax_all'].to_csv(path_data+f'sensitivity_tuning/{safe}_relaxed_all.csv',index=False)
all_performance = pd.concat(summaries,ignore_index=True); all_predictions = pd.concat(predictions_all,ignore_index=True)
all_performance.to_csv(path_data+'all_9case_performance_long.csv',index=False); all_predictions.to_csv(path_data+'all_9case_predictions.csv',index=False)
primary = runs[('S2',0)]

#add
# ============================================================
# SUPPLEMENTARY AUTOCORRELATION + RESIDUAL NONLINEARITY DIAGNOSTICS
# Primary S2, lag 0, using one-step-ahead CV residuals (2006-2015)
# ============================================================
diag_rows=[]
for model,key in [('OLS','cvpred_ols'),('LASSO','cvpred_lasso'),('Relaxed LASSO','cvpred_relax')]:
    d=residual_diagnostics(primary[key]); d['Model']=model; diag_rows.append(d)
residual_diagnostics_table=pd.DataFrame(diag_rows)[['Model','N','Residual_Mean','Residual_SD','Lag1_Residual_ACF']]
residual_diagnostics_table.to_csv(path_data+'Supplement_residual_autocorrelation.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(9.2,3.2),sharey=True)
for ax,(model,key) in zip(axes,[('OLS','cvpred_ols'),('LASSO','cvpred_lasso'),('Relaxed LASSO','cvpred_relax')]):
    df=primary[key]; fitted=df['Predicted'].to_numpy(dtype=float); resid=df['Observed'].to_numpy(dtype=float)-fitted
    ax.scatter(fitted,resid,s=28); ax.axhline(0,linestyle='--',linewidth=1)
    ax.set_title(model,fontsize=9); ax.set_xlabel('One-step-ahead prediction',fontsize=8); ax.tick_params(axis='both',labelsize=8); ax.grid(alpha=.2)
axes[0].set_ylabel('Prediction residual',fontsize=8)
plt.tight_layout(); plt.savefig(path_data+'Supplement_residual_vs_predicted.png',dpi=300,bbox_inches='tight'); plt.close()

# ============================================================
# TABLE 2: PRIMARY MIN-CV VS 1-SE
# ============================================================
def tuning_row(method,rule,row):
    return {'Method':method,'Rule':rule,'lambda':float(row['lambda']),'gamma':np.nan if method=='LASSO' else float(row['gamma']),'NNZ':int(row['NNZ']),'CV_MSE':float(row['CV_MSE']),'CV_RMSE':float(row['CV_RMSE']),'Fold_SE_MSE':float(row['Fold_SE_MSE'])}
table2 = pd.DataFrame([
    tuning_row('LASSO','Minimum CV',primary['lasso_min']),tuning_row('LASSO','1-SE',primary['lasso_1se']),
    tuning_row('Relaxed LASSO','Minimum CV',primary['relax_min']),tuning_row('Relaxed LASSO','1-SE',primary['relax_1se'])])
# Attach selected variables from models fitted at these tuning values
sel = []
for _,r in table2.iterrows():
    if r['Method']=='LASSO': m=fit_lasso(X[years<=2015],y[years<=2015],r['lambda'],tol)
    else: m=fit_relaxed_lasso(X[years<=2015],y[years<=2015],r['lambda'],r['gamma'],tol)
    sel.append(selected_feature_names(m['beta'],features))
table2['Selected_predictors']=sel; table2=table2[['Method','Rule','lambda','gamma','NNZ','Selected_predictors','CV_MSE','CV_RMSE','Fold_SE_MSE']]
table2.to_csv(path_data+'Table2_primary_tuning.csv',index=False)
primary['lasso_by_nnz'].to_csv(path_data+'Supplement_S2_lasso_best_by_nnz.csv',index=False); primary['relax_by_nnz'].to_csv(path_data+'Supplement_S2_relaxed_best_by_nnz.csv',index=False)

# Primary coefficient table (1-SE)
coef_table = pd.DataFrame({'Predictor':features,'LASSO':primary['models']['LASSO']['beta'],'Relaxed_LASSO':primary['models']['Relaxed LASSO']['beta']})
coef_table.to_csv(path_data+'Primary_1SE_coefficients.csv',index=False)

# FIGURE 5 coefficient path using primary lambda grid
X_primary = X[years<=2015]; y_primary = y[years<=2015]; path = np.zeros((len(primary['lambda_grid']),len(features)))
for i,comp in enumerate(primary['full_path']): path[i,:]=model_from_components(comp,1.0)['beta']
plt.figure(figsize=(5.8,3.6))
for j,name in enumerate(features): plt.plot(np.log10(primary['lambda_grid']),path[:,j],label=name)
plt.axvline(np.log10(float(primary['lasso_1se']['lambda'])),linestyle='--',linewidth=1.2,label=r'1-SE selected $\lambda$'); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel('LASSO coefficient'); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(path_data+'Fig5_lasso_coefficient_path.png',dpi=300,bbox_inches='tight'); plt.close()

# ============================================================
# TABLE 3 + FIGURE 6 : PRIMARY OUT-OF-SAMPLE PERFORMANCE
# ============================================================
table3 = primary['summary'][['Model','NNZ','Selected_predictors','CV_RMSE','CV_MAE','Test_RMSE','Test_MAE']].copy()
table3.to_csv(path_data+'Table3_primary_test_performance.csv',index=False)
primary['predictions'].to_csv(path_data+'primary_test_predictions.csv',index=False)
pp = primary['predictions']; plt.figure(figsize=(5.8,3.6)); plt.plot(pp['Year'],pp['Observed'],marker='o',linewidth=2,label='Observed'); plt.plot(pp['Year'],pp['OLS'],marker='o',linestyle='-.',label='OLS')
plt.plot(pp['Year'],pp['LASSO'],marker='o',linestyle='--',label='LASSO'); plt.plot(pp['Year'],pp['Relaxed_LASSO'],marker='o',linestyle=':',label='Relaxed LASSO')
plt.xlabel('Year'); plt.ylabel('Air-pollution-related death rate'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(path_data+'Fig6_primary_observed_vs_predicted.png',dpi=300,bbox_inches='tight'); plt.close()

# ============================================================
# TABLE 4 + FIGURE 7: PREDICTIVE ROBUSTNESS ACROSS 9 CASES
# ============================================================
table4=all_performance[['Case','Split','Lag','Model','lambda','gamma','NNZ','Selected_predictors','CV_RMSE','CV_MAE','Test_RMSE','Test_MAE']].copy()
table4.to_csv(path_data+'Table4_all_9case_predictive_performance.csv',index=False)
# In-sample fit is calculated after the selected tuning parameters are refitted on the full training period; it is not CV performance.
train_fit_table=all_performance[['Case','Split','Lag','Model','NNZ','Train_RMSE','Train_MAE']].copy()
train_fit_table.to_csv(path_data+'Supplement_S3_training_fit_selected_models.csv',index=False)

# Robustness summary: means, medians, and number of lowest-RMSE cases
robust_rows=[]
for model in ['OLS','LASSO','Relaxed LASSO']:
    s=all_performance[all_performance['Model']==model]; robust_rows.append({'Model':model,'Mean_RMSE':s['Test_RMSE'].mean(),'Median_RMSE':s['Test_RMSE'].median(),'Mean_MAE':s['Test_MAE'].mean(),'Median_MAE':s['Test_MAE'].median()})
robust_summary=pd.DataFrame(robust_rows)
wins={'OLS':0,'LASSO':0,'Relaxed LASSO':0}
for case,sub in all_performance.groupby('Case'):
    wins[sub.loc[sub['Test_RMSE'].idxmin(),'Model']]+=1
robust_summary['Lowest_RMSE_Cases']=robust_summary['Model'].map(wins); robust_summary.to_csv(path_data+'Table4_summary_across_9cases.csv',index=False)

# FIGURE 7: three split panels; model=color, lag=line style
colors={'OLS':'tab:blue','LASSO':'tab:orange','Relaxed LASSO':'tab:green'}; styles={0:'-',1:'--',2:':'}; model_cols={'OLS':'OLS','LASSO':'LASSO','Relaxed LASSO':'Relaxed_LASSO'}
fig,axes=plt.subplots(1,3,figsize=(15,4.6),sharey=True)
for ax,split in zip(axes,splits):
    # observed series is identical across lags for a given split; plot once from lag 0
    p0=runs[(split['id'],0)]['predictions']; ax.plot(p0['Year'],p0['Observed'],color='black',marker='o',linewidth=2.3,label='Observed')
    for model in ['OLS','LASSO','Relaxed LASSO']:
        for lag in lags:
            p=runs[(split['id'],lag)]['predictions']; ax.plot(p['Year'],p[model_cols[model]],color=colors[model],linestyle=styles[lag],linewidth=1.4,alpha=.9)
    # Force annual integer ticks and add horizontal margins
    test_years=np.sort(p0['Year'].unique()).astype(int)
    ax.set_xticks(test_years)
    ax.set_xlim(test_years.min()-0.3,test_years.max()+0.3)
    ax.set_title(split['label']); ax.set_xlabel('Response year'); ax.grid(alpha=.2)
axes[0].set_ylabel('Air-pollution-related death rate')
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],color='black',marker='o',lw=2,label='Observed')]+[Line2D([0],[0],color=colors[m],lw=2,label=m) for m in colors]+[Line2D([0],[0],color='gray',lw=2,linestyle=styles[l],label=f'Lag {l}') for l in lags]
fig.legend(handles=handles,loc='upper center',ncol=7,frameon=True,bbox_to_anchor=(.5,1.03)); plt.tight_layout(rect=[0,0,1,.91])
plt.savefig(path_data+'Fig7_all_sensitivity_observed_vs_predicted.png',dpi=300,bbox_inches='tight'); plt.close()

# ============================================================
# TABLE 5 + FIGURE 8: FEATURE-SELECTION STABILITY ACROSS 9 CASES
# ============================================================
selection_rows=[]
for (split_id,lag),run in runs.items():
    for model in ['LASSO','Relaxed LASSO']:
        row={'Case':run['case'],'Split':run['split']['label'],'Lag':lag,'Model':model}; beta=run['betas'][model]
        for j,name in enumerate(features): row[name]=int(abs(beta[j])>1e-10)
        selection_rows.append(row)
selection_matrix=pd.DataFrame(selection_rows); selection_matrix.to_csv(path_data+'Supplement_S4_selection_matrix.csv',index=False)
rank_rows=[]
for model in ['LASSO','Relaxed LASSO']:
    sub=selection_matrix[selection_matrix['Model']==model]
    for name in features:
        count=int(sub[name].sum()); rank_rows.append({'Model':model,'Predictor':name,'Selection_Count':count,'Selection_Frequency':count/9})
rank_long=pd.DataFrame(rank_rows)
rank_wide=rank_long.pivot(index='Predictor',columns='Model',values=['Selection_Count','Selection_Frequency']).reset_index()
rank_wide.columns=['Predictor','LASSO_Count','Relaxed_Count','LASSO_Frequency','Relaxed_Frequency']
# pandas pivot may order model levels alphabetically; enforce by recomputing safely
rank_wide=pd.DataFrame({'Predictor':features})
for model,prefix in [('LASSO','LASSO'),('Relaxed LASSO','Relaxed')]:
    tmp=rank_long[rank_long['Model']==model].set_index('Predictor'); rank_wide[prefix+'_Count']=rank_wide['Predictor'].map(tmp['Selection_Count'])
    rank_wide[prefix+'_Frequency']=rank_wide['Predictor'].map(tmp['Selection_Frequency'])
rank_wide['Average_Frequency']=(rank_wide['LASSO_Frequency']+rank_wide['Relaxed_Frequency'])/2; rank_wide=rank_wide.sort_values(['Average_Frequency','Predictor'],ascending=[False,True]).reset_index(drop=True)
rank_wide.insert(0,'Rank',np.arange(1,len(rank_wide)+1)); table5=rank_wide[['Rank','Predictor','LASSO_Count','LASSO_Frequency','Relaxed_Count','Relaxed_Frequency','Average_Frequency']]
table5.to_csv(path_data+'Table5_feature_selection_stability.csv',index=False)
plot_rank=table5.sort_values('Average_Frequency',ascending=True); ypos=np.arange(len(plot_rank)); h=.35; 
fig,ax=plt.subplots(figsize=(5.8,3.6))
b1=ax.barh(ypos-h/2,100*plot_rank['LASSO_Frequency'],height=h,label='LASSO')
b2=ax.barh(ypos+h/2,100*plot_rank['Relaxed_Frequency'],height=h,label='Relaxed LASSO'); ax.set_yticks(ypos); ax.set_yticklabels(plot_rank['Predictor'])
ax.set_xlabel('Selection frequency across nine temporal specifications (%)'); ax.set_xlim(0,110); ax.set_xticks(np.arange(0,101,20)); ax.bar_label(b1,fmt='%.0f%%',padding=3,fontsize=8) 
ax.bar_label(b2,fmt='%.0f%%',padding=3,fontsize=8); ax.legend(fontsize=8); ax.grid(axis='x',alpha=.2); plt.tight_layout(); plt.savefig(path_data+'Fig8_feature_selection_stability.png',dpi=300,bbox_inches='tight'); plt.close()

# ============================================================
# PRINT ONLY MANUSCRIPT / SUPPLEMENT RESULTS
# ============================================================
pd.set_option('display.max_columns',50); pd.set_option('display.width',220)
print('\nDATA SUMMARY')
print(f'Observations: 1990-2021 (n={len(years)}); primary training: 1990-2015 (n={np.sum(years<=2015)}); primary test: 2016-2021 (n={np.sum(years>=2016)})')
print('\nTABLE 1. VIF')
print(vif_table.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
#----------------------------------------------------------------------------------
#add
print('\nSUPPLEMENTARY NONLINEARITY DIAGNOSTIC: LINEAR VS QUADRATIC MARGINAL FIT')
print(nonlinearity_table.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nSUPPLEMENTARY AUTOCORRELATION DIAGNOSTIC: PRIMARY ONE-STEP-AHEAD CV RESIDUALS')
print(residual_diagnostics_table.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
#----------------------------------------------------------------------------------
print('\nTABLE 2. PRIMARY TEMPORAL MODEL SELECTION')
print(table2.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nPRIMARY 1-SE COEFFICIENTS (original reporting scale: per 10^5 tonnes)')
print(coef_table.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nTABLE 3. PRIMARY OUT-OF-SAMPLE PERFORMANCE')
print(table3.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nTABLE 4. PREDICTIVE PERFORMANCE ACROSS 3 SPLITS x 3 LAGS')
print(table4.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nTABLE 4 SUMMARY. PERFORMANCE ACROSS NINE CASES')
print(robust_summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('\nTABLE 5. FEATURE-SELECTION STABILITY ACROSS NINE 1-SE SPECIFICATIONS')
print(table5.to_string(index=False,formatters={'LASSO_Frequency':lambda x:f'{100*x:.1f}%','Relaxed_Frequency':lambda x:f'{100*x:.1f}%','Average_Frequency':lambda x:f'{100*x:.1f}%'}))


end=time.time(); print(f'\nTotal runtime: {end-begin:.2f} seconds')