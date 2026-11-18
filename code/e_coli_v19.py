"""E. coli model development and retrospective evaluation.

Model selection uses 2018–2023 historical predictions. The 2024–2026
period was inspected in earlier research versions and is not an independent
prospective test. Weather inputs are aligned with sampling dates. The
laboratory-history adjustment assumes a 28-day reporting lag that has not
been verified against publication timestamps.
"""
from pathlib import Path
from datetime import timedelta
import hashlib, json, warnings, time
import numpy as np, pandas as pd
from scipy.special import expit, logit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SEED=42
np.random.seed(SEED)
DATA_FILENAME = 'Aquacast20Years_Weekly_UPDATED_TO_2026_10(2).csv'
INPUT_CANDIDATES = [Path.cwd()/DATA_FILENAME, Path.cwd()/'data'/DATA_FILENAME,
                    Path('/mnt/data')/DATA_FILENAME, Path('/content')/DATA_FILENAME,
                    Path('/content/drive/MyDrive/Datasets')/DATA_FILENAME]
INPUT = next((p for p in INPUT_CANDIDATES if p.is_file()), None)
if INPUT is None:
    raise FileNotFoundError('Place '+DATA_FILENAME+' in Colab /content, Drive MyDrive/Datasets, or working directory.')
OUT=Path.cwd()/'outputs'/'e_coli'/'reproduced_v19'; OUT.mkdir(parents=True,exist_ok=True)
FILE_SHA=hashlib.sha256(INPUT.read_bytes()).hexdigest()
assert FILE_SHA=='6ee536b6e0051d087277726ae02ecc7d2bf23960752febc2160e6f884aa52485', 'Unexpected input. Check first.'
full=pd.read_csv(INPUT,low_memory=False)
assert len(full)==1890 and full.shape[1]==137
D=full[full.indicator_clean=='E. coli'].copy()
D['sample_date']=pd.to_datetime(D.sample_date)
D=D.sort_values('sample_date').reset_index(drop=True)
assert len(D)==955 and D.sample_date.max()==pd.Timestamp('2026-10-19')
D['actual']=(pd.to_numeric(D.result_value,errors='raise')>=400).astype(int)
assert int(D.actual.sum())==283
# mask incomplete feature windows (this is not imputing bacteria labels)
windows={
 'rain_3day_data_days':(3,['rain_3day_sum','rain_3day_avg','rain_3day_max','rain_3day_rainy_days','rain_intensity_3day','rain_ratio_1to3','wet_season_rain_3day_sum','rain_temp_interaction_3day']),
 'rain_7day_data_days':(7,['rain_7day_sum','rain_7day_avg','rain_7day_max','rain_7day_rainy_days','rain_intensity_7day','wet_season_rain_7day_sum']),
 'rain_14day_data_days':(14,['rain_14day_sum','rain_14day_avg','rain_14day_max','rain_14day_rainy_days','rain_intensity_14day']),
 'temp_7day_data_days':(7,['temp_7day_avg','temp_7day_min','temp_7day_max']),
 'temp_14day_data_days':(14,['temp_14day_avg','temp_14day_min','temp_14day_max'])}
for col,(n,features) in windows.items():
 v=pd.to_numeric(D[col],errors='coerce')
 bad=v.isna()|(v<n)
 D.loc[bad,[f for f in features if f in D.columns]]=np.nan
D['month_sin']=np.sin(2*np.pi*(D.sample_date.dt.dayofyear-1)/365.25)
D['month_cos']=np.cos(2*np.pi*(D.sample_date.dt.dayofyear-1)/365.25)
rain=['rain_1day','rain_3day_sum','rain_1day_lag1','rain_1day_lag2','rain_ratio_1to3','adp_days','first_flush_index']
temp=['temp_3day_avg','temp_7day_avg','temp_14day_avg']
season=['season_wet','month_sin','month_cos']
for f in ['rain_1day','rain_3day_sum','rain_7day_sum','rain_14day_sum','rain_3day_max','rain_7day_max','rain_1day_lag1','rain_1day_lag2','first_flush_index','adp_days']:
 v=pd.to_numeric(D[f],errors='coerce')
 D['log_'+f]=np.log1p(v.clip(lower=0))
lograin=['log_'+f for f in ['rain_1day','rain_3day_sum','rain_7day_sum','rain_14day_sum','rain_3day_max','rain_7day_max','rain_1day_lag1','rain_1day_lag2','first_flush_index','adp_days']]
FSETS={
 'rain_temp':rain+temp,
 'rain_temp_season':rain+temp+season,
 'lograin_season':lograin+season,
 'rain_season':rain+season,
 'hybrid_rain':rain+lograin+temp+season,
}
for f in set(z for ll in FSETS.values() for z in ll):
 D[f]=pd.to_numeric(D[f],errors='coerce').replace([np.inf,-np.inf],np.nan)
# Main comparisons: no unverified bacterial lab-history features in estimator.
MODELS={
 'svm_smooth':lambda:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),SVC(kernel='rbf',C=.3,gamma='scale',class_weight='balanced',probability=True,random_state=SEED)),
 'svm_mid':lambda:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),SVC(kernel='rbf',C=2,gamma='scale',class_weight='balanced',probability=True,random_state=SEED)),
 'svm_narrow':lambda:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),SVC(kernel='rbf',C=1,gamma=.04,class_weight='balanced',probability=True,random_state=SEED)),
 'rf':lambda:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RandomForestClassifier(n_estimators=160,min_samples_leaf=4,class_weight='balanced_subsample',max_features=.8,random_state=SEED,n_jobs=1)),
 'extra':lambda:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),ExtraTreesClassifier(n_estimators=160,min_samples_leaf=4,class_weight='balanced_subsample',max_features=.8,random_state=SEED,n_jobs=1)),
 'lr':lambda:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.15,class_weight='balanced',max_iter=1500,random_state=SEED)),
}
YEARS=[(2018,2019),(2020,2021),(2022,2023)]
DATE=D.sample_date
ACT=D.actual.to_numpy(dtype=int)

def score(y,p,cut):
 pr=np.asarray(p)>=cut
 tp=int(np.sum((y==1)&pr));fp=int(np.sum((y==0)&pr));fn=int(np.sum((y==1)&(~pr)));tn=int(np.sum((y==0)&(~pr)))
 prec=tp/(tp+fp) if tp+fp else 0.
 rec=tp/(tp+fn) if tp+fn else 0.
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=prec,recall=rec,specificity=tn/(tn+fp) if tn+fp else np.nan)

def historical_cutoff(y,p,floor=.50):
 # Efficient, exact ranking on validation only, no test outcomes.
 ind=np.argsort(-p,kind='mergesort'); yy=y[ind]
 tp=np.cumsum(yy); fp=np.cumsum(1-yy)
 possible=(tp/(tp+fp)>=floor)&(fp <= max(0,int((len(y)-y.sum())*.95)))
 unique_end=np.r_[p[ind][:-1]!=p[ind][1:],True]
 possible &= unique_end
 if possible.any():
  ix=np.where(possible)[0]
  # maximize recall, then precision, then prefer fewer positives
  z=sorted(ix,key=lambda j:(int(tp[j]),float(tp[j]/(tp[j]+fp[j])),-int(j)),reverse=True)[0]
  return float(p[ind][z]),True
 # If minimum precision unattainable: use historical F2, clearly flagged.
 f2=5*tp/(5*tp+4*(int(y.sum())-tp)+fp+1e-12)
 z=int(np.argmax(np.where(unique_end,f2,-1)))
 return float(p[ind][z]),False

# Lag-28 diagnostic: assuming all laboratory results are published within 28 days.
# Does not create same-day or future label leakage. Research only until verified.
LAB_LAG_DAYS=28
DAYS=(DATE.to_numpy(dtype='datetime64[ns]'))
# OOF feature: earliest index with specimen date > querydate-28 days,
# so all referenced laboratory observations are at least 28 days old.
PAST_END=np.searchsorted(DAYS,DAYS-np.timedelta64(LAB_LAG_DAYS,'D'),side='right')
assert np.all((PAST_END<=np.arange(len(D))))

# Optional adaptive probability odds correction based on *past* observed lab outcomes.
# The threshold and correction strength are learned from pre-2024 CV only.
# No lab features used in estimator, and never uses same or recent 28-day labels.
def recent_rate(i,months,prior):
 end=int(PAST_END[i]); cutoff=DAYS[i]-np.timedelta64(months*30,'D')
 beg=int(np.searchsorted(DAYS,cutoff,side='left'))
 beg=min(beg,end)
 y=ACT[beg:end]
 # Beta-style shrinkage to training prevalence avoids unstable tiny sample rates.
 shrink=8.
 return (float(y.sum())+shrink*prior)/(len(y)+shrink)

def adjusted_scores(pp,indices,trainprev,gamma,months):
 if gamma==0:return np.asarray(pp,dtype=float)
 # Odds correction requires calibration; this is a prespecified research strategy.
 obs=np.array([recent_rate(int(i),months,float(trainprev[j])) for j,i in enumerate(indices)])
 base=np.clip(trainprev,1e-5,1-1e-5)
 adj=expit(logit(np.clip(pp,1e-6,1-1e-6)) + gamma*(logit(np.clip(obs,1e-5,1-1e-5))-logit(base)))
 return np.asarray(adj)

MASKDEV=(DATE.dt.year.between(2018,2023)).to_numpy()
MASKTEST=(DATE.dt.year>=2024).to_numpy()
assert MASKDEV.sum()==232 and MASKTEST.sum()==144
Xcache={k:D[v].to_numpy(dtype=float) for k,v in FSETS.items()}

# Past lab results strictly lagged by 28 days: no same-week or future sampling data.
LAB=[]
for i in range(len(D)):
 end=int(PAST_END[i]); prev=ACT[:end]; d=DAYS[:end]
 rec={'last_lab_exceed': float(prev[-1]) if end else np.nan,
      'days_since_lab':float((DAYS[i]-d[-1])/np.timedelta64(1,'D')) if end else np.nan,
      'days_since_exceed':float((DAYS[i]-d[np.where(prev==1)[0][-1]])/np.timedelta64(1,'D')) if np.any(prev==1) else np.nan}
 for period in [90,180,365,730]:
  idx=int(np.searchsorted(d,DAYS[i]-np.timedelta64(period,'D'),side='left'))
  yy=prev[idx:]; n=len(yy)
  rec[f'lab_n_{period}']=float(n)
  rec[f'lab_positive_{period}']=float(yy.sum())
  rec[f'lab_prevalence_{period}']=(float(yy.sum())+2)/(n+8)
 LAB.append(rec)
L=pd.DataFrame(LAB)
assert (DAYS[:1]<DAYS[-1:]).all()
lab_cols=list(L.columns)
for c in lab_cols:D[c]=L[c]
FSETS.update({
 'laglab_rainseason': FSETS['lograin_season']+lab_cols,
 'laglab_hybrid': FSETS['hybrid_rain']+lab_cols,
 'laglab_rain_temp': FSETS['rain_temp_season']+lab_cols,
 'laglab_only':lab_cols+['month_sin','month_cos'],
})
Xcache={k:D[v].to_numpy(dtype=float) for k,v in FSETS.items()}

def make_model(name):
 if name=='extra_a':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True), ExtraTreesClassifier(n_estimators=100,min_samples_leaf=4,max_features=.8,class_weight='balanced_subsample',random_state=SEED,n_jobs=1))
 if name=='extra_b':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True), ExtraTreesClassifier(n_estimators=100,min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample',random_state=SEED,n_jobs=1))
 if name=='rf':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True), RandomForestClassifier(n_estimators=100,min_samples_leaf=3,class_weight='balanced_subsample',max_features=.7,random_state=SEED,n_jobs=1))
 if name=='logit':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(), LogisticRegression(C=.25,class_weight='balanced',max_iter=1800,random_state=SEED))
 if name=='svm':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True), StandardScaler(), SVC(kernel='rbf',C=.8,gamma='scale',class_weight='balanced',probability=True,random_state=SEED))
 raise KeyError(name)

YEARS=[(2018,2019),(2020,2021),(2022,2023)]
TEST=[2024,2025,2026]
CANDS=[];DEV_P=[];EVAL_P=[]
win_opts=[None,5,9]
feat_opts=['lograin_season','hybrid_rain','rain_temp_season','laglab_rainseason','laglab_hybrid','laglab_rain_temp','laglab_only']
name_opts=['extra_a','extra_b','rf','logit','svm']
print('CANDIDATES',len(feat_opts)*len(name_opts)*len(win_opts),flush=True)
now=time.time(); n=0
for features in feat_opts:
 X=Xcache[features]
 for model_name in name_opts:
  for window in win_opts:
   dev_parts=[];test_parts=[]
   for start,end in YEARS+[(y,y) for y in TEST]:
    fit=(DATE.dt.year<start).to_numpy()
    if window is not None:fit &= (DATE.dt.year>=start-window).to_numpy()
    val=DATE.dt.year.between(start,end).to_numpy()
    if fit.sum()<110 or ACT[fit].sum()<10:
     dev_parts=[];break
    try:
     with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      mod=make_model(model_name);mod.fit(X[fit],ACT[fit]);p=mod.predict_proba(X[val])[:,1]
    except Exception as e:
     print('MODEL ERROR',str(e)[:200],features,model_name,window,flush=True);dev_parts=[];break
    fr=pd.DataFrame({'idx':np.flatnonzero(val),'date':DATE[val].values,'year':DATE[val].dt.year.values,'actual':ACT[val],'raw':p,'train_prev':float(ACT[fit].mean())})
    if end<2024:dev_parts.append(fr)
    else:test_parts.append(fr)
   if len(dev_parts)!=3 or len(test_parts)!=3:continue
   dev=pd.concat(dev_parts,ignore_index=True);test=pd.concat(test_parts,ignore_index=True)
   if len(dev)!=232 or len(test)!=144:raise ValueError('Bad time split')
   for adjust in ['none','odds_3m','odds_6m','odds_12m','odds_3mstrong','blend_3m6m']:
    def adapt(df,method):
     pp=df.raw.to_numpy();inds=df.idx.to_numpy(dtype=int);tr=df.train_prev.to_numpy()
     if method=='none':return pp
     if method=='odds_3m':return adjusted_scores(pp,inds,tr,1.0,3)
     if method=='odds_6m':return adjusted_scores(pp,inds,tr,1.0,6)
     if method=='odds_12m':return adjusted_scores(pp,inds,tr,1.0,12)
     if method=='odds_3mstrong':return adjusted_scores(pp,inds,tr,1.5,3)
     if method=='blend_3m6m':return (adjusted_scores(pp,inds,tr,1.0,3)+adjusted_scores(pp,inds,tr,1.0,6))/2
    dp=adapt(dev,adjust);tp=adapt(test,adjust)
    cutoff,qual=historical_cutoff(dev.actual.to_numpy(),dp,.50)
    dm=score(dev.actual.to_numpy(),dp,cutoff)
    tm=score(test.actual.to_numpy(),tp,cutoff)
    key=f'{features}|{model_name}|{window}|{adjust}'
    CANDS.append(dict(key=key,features=features,model=model_name,window=str(window),adjustment=adjust,cutoff=cutoff,qualified=qual,
     cv_precision=dm['precision'],cv_recall=dm['recall'],cv_tp=dm['tp'],cv_fp=dm['fp'],cv_fn=dm['fn'],cv_auc=roc_auc_score(dev.actual,dp),cv_ap=average_precision_score(dev.actual,dp),
     test_precision=tm['precision'],test_recall=tm['recall'],test_tp=tm['tp'],test_fp=tm['fp'],test_fn=tm['fn'],test_tn=tm['tn'],
     retrospective_target_met=tm['recall']>.70 and tm['precision']>.50,
     lagged_lab_features=features.startswith('laglab')))
    DEV_P.append((key,dev,dp))
    EVAL_P.append((key,test,tp))
   n+=1
   if n%20==0:print('FIT',n,'configurations',round(time.time()-now,1),'sec',flush=True)
r=pd.DataFrame(CANDS)
# Model selection is on DEV metrics alone, logged alongside retrospective results.
r['cv_strict50']=(r.cv_precision>.5)
r['cv_recall70']=(r.cv_recall>.7)
ranked=r.sort_values(['cv_strict50','cv_recall','cv_precision','cv_ap'],ascending=[False,False,False,False])
r.to_csv(OUT/'all_historical_tuned_candidates_with_retrospective_metrics.csv',index=False)
ranked.to_csv(OUT/'pre2024_ranked_all_candidates.csv',index=False)
# Top prespecified-type leader and top no-lab candidate by historical ranking
locked=ranked.iloc[0]
no_lab=ranked[~ranked.lagged_lab_features].iloc[0]
print('TOP 10 HISTORICAL (not selected by 2024):\n',ranked.head(10)[['key','cv_recall','cv_precision','test_recall','test_precision','test_tp','test_fp']].to_string(index=False),flush=True)
print('TOP NO LAB FEATURES (may use LAB ODDS adjustment):\n',no_lab[['key','cv_recall','cv_precision','test_recall','test_precision']].to_string(),flush=True)
print('RETRO CANDIDATES NUMERIC PASS',int(r.retrospective_target_met.sum()),'of',len(r),flush=True)
# Export prediction rows for historically rank1 and no-lab historically rank1.
for label,win in [('historical_rank1',locked),('historical_no_lab_features_rank1',no_lab)]:
 key=win['key'];df=next((t.copy() for k,t,p in EVAL_P if k==key));prob=next(p for k,t,p in EVAL_P if k==key)
 df['probability']=prob;df['cutoff']=win.cutoff;df['predicted']=(prob>=win.cutoff).astype(int);df['candidate']=key
 df.to_csv(OUT/(label+'_2024plus_predictions.csv'),index=False)
 print(label,df.groupby('year').apply(lambda v:pd.Series(score(v.actual.to_numpy(),v.probability.to_numpy(),win.cutoff)),include_groups=False).to_string(),flush=True)
# All purely descriptive later metrics; scan is not legitimate for final model picking.
r.sort_values(['retrospective_target_met','test_recall','test_precision'],ascending=[False,False,False]).head(50).to_csv(OUT/'POSTHOC_TOP50_FOR_DIAGNOSTICS_ONLY.csv',index=False)
(OUT/'README.md').write_text('''E. coli V19 research. 955 observed water samples, 400 MPN/100 mL.\nHistorical OOF dev 2018-2023; retrospectively examined 2024-2026: NO test-based selection.\nHistorical ranking is exploratory because many choices share same historical OOF labels.\nLagged-laboratory features or probability adaptation assume all results available 28 days after sample; this is NOT verified from posting timestamps.\nUse of same-day observed weather supports nowcast experiments, NOT seven-day-ahead forecast claims.\nPrior examination of 2024-2026 outcomes means test is NOT virgin independent validation.\nThe model quality may or may not reach the numerical target. NEVER choose candidate based on POSTHOC table for an independent-performance claim.\n''')
(OUT/'manifest.json').write_text(json.dumps({'input_sha256':FILE_SHA,'n_ecoli':len(D),'target':'>70% recall and >50% precision','development_period':'2018-2023','evaluation_period':'2024-2026 (previously inspected)','time_aware_lab_availability_assumption_days':28,'candidate_count':len(r)},indent=2))
print('DONE',OUT,round(time.time()-now,1),flush=True)

