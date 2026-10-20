import numpy as np, warnings, json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")
LB=np.array([4.8,6,0,3.]); UB=np.array([8.8,14,0.2,7.])
def coded(X): return (np.asarray(X)-(LB+UB)/2)/((UB-LB)/4)
def rsm(X):
    x1,x2,x3,x4=coded(X).T
    return (37.172+2.425*x1+2.129*x2+1.318*x3+2.230*x4-2.562*x1**2-1.930*x2**2-1.313*x3**2-2.114*x4**2+1.519*x1*x2+0.759*x2*x4)
def met(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float); e=y-p
    return dict(n=len(y),RMSE=np.sqrt(np.mean(e**2)),MAE=np.mean(abs(e)),bias=e.mean(),
        R2=1-np.sum(e**2)/np.sum((y-y.mean())**2),AAD=100*np.mean(abs(e)/y),MPE=100*np.mean(e/y),
        TIC=np.sqrt(np.mean(e**2))/(np.sqrt(np.mean(y**2))+np.sqrt(np.mean(p**2))),r=np.corrcoef(y,p)[0,1])
# CCD (Table S2)
ccd=np.array([l.split() for l in """7.8 12 0.15 4 31.922 33.65
6.8 10 0.10 5 35.737 37.17
6.8 10 0.10 5 38.256 37.17
6.8 10 0.10 3 24.307 24.26
6.8 10 0.10 5 36.682 37.17
7.8 8 0.05 4 25.401 25.24
8.8 10 0.10 5 32.236 31.78
5.8 8 0.15 4 25.888 26.06
7.8 8 0.05 6 27.260 28.18
5.8 12 0.15 4 26.527 25.77
6.8 10 0.10 5 38.410 37.17
6.8 10 0.20 5 35.017 34.55
6.8 10 0.10 5 37.854 37.17
5.8 8 0.05 6 25.304 26.37
7.8 8 0.15 6 30.683 30.82
5.8 12 0.05 4 23.249 23.13
6.8 6 0.10 5 26.142 25.19
6.8 10 0.10 7 34.175 33.18
6.8 10 0 5 29.866 29.28
4.8 10 0.10 5 22.662 22.08
6.8 10 0.10 5 36.090 37.17
5.8 8 0.05 4 22.957 23.43
7.8 12 0.05 6 37.836 37.00
5.8 12 0.15 6 31.752 31.75
7.8 12 0.15 6 39.261 39.63
6.8 14 0.10 5 33.811 33.71
5.8 8 0.15 6 29.189 29.01
5.8 12 0.05 6 27.539 29.11
7.8 8 0.15 4 27.370 27.88
7.8 12 0.05 4 31.726 31.02""".splitlines()],float)
Xc,yc,pc=ccd[:,:4],ccd[:,4],ccd[:,5]
bo=np.array([l.split() for l in """6.42 8.50 0.139 6.78 30.11
5.78 11.89 0.106 5.54 32.89
7.85 13.84 0.151 3.93 31.04
6.06 6.55 0.034 5.37 23.12
6.97 10.23 0.081 6.40 37.31
8.63 7.64 0.067 4.57 22.10
5.05 12.09 0.182 4.43 19.15
7.43 9.13 0.013 3.26 18.99
6.83 11.14 0.093 6.66 37.11
6.79 6.00 0.091 3.00 15.47
7.61 13.98 0.000 6.08 28.92
8.66 12.38 0.096 6.54 36.15
7.31 10.95 0.070 7.00 34.17
7.46 11.99 0.095 6.17 40.04
7.60 14.00 0.114 5.93 37.92
7.70 11.95 0.119 6.00 40.07
8.80 14.00 0.200 6.01 34.09
7.43 12.29 0.111 6.11 41.21
7.26 12.68 0.134 6.28 40.51
7.25 12.38 0.118 6.12 39.73""".splitlines()],float)
Xb,yb=bo[:,:4],bo[:,4]
rsm_bo_tab=np.array([30.12,31.97,30.54,22.52,36.17,22.14,18.64,20.00,36.24,15.04,30.18,35.68,34.15,39.32,38.36,40.02,33.40,39.75,39.05,39.52])
Xe=np.array([[8.30,7.54,0.147,5.37],[7.92,11.86,0.169,4.97],[5.02,9.95,0.090,3.86],[5.51,13.41,0.007,3.30],[6.08,12.35,0.036,6.10],[7.10,6.05,0.186,6.82],[6.74,8.73,0.121,4.01],[7.64,10.91,0.061,5.65]])
ye=np.array([27.59,35.11,18.13,19.23,31.11,24.21,23.13,36.07])
rsm_e=np.array([26.94,37.82,19.12,5.37,29.73,17.73,31.48,37.54])
gp_e=np.array([27.90,32.99,17.34,23.58,33.69,26.53,18.75,35.78])
def u(X): return (X-LB)/(UB-LB)
def gp_fit(X,y,seed=0):
    k=C(1.0,(1e-2,1e3))*Matern(length_scale=np.ones(4),length_scale_bounds=(1e-2,1e1),nu=2.5)+WhiteKernel(1e-2,(1e-3,1e1))
    return GaussianProcessRegressor(k,normalize_y=True,n_restarts_optimizer=12,random_state=seed).fit(u(X),y)
if __name__=="__main__":
    print("check rsm eq vs table S6 max diff", np.max(abs(rsm(Xb)-rsm_bo_tab)), "ext", np.max(abs(rsm(Xe)-rsm_e)))
    out={}
    out['RSM fit, 30 CCD (training)']=met(yc,pc)
    out['RSM external, 20 BO']=met(yb,rsm_bo_tab)
    loo=np.array([gp_fit(np.delete(Xb,i,0),np.delete(yb,i)).predict(u(Xb[i:i+1]))[0] for i in range(20)])
    out['GP LOO, 20 BO']=met(yb,loo)
    g20=gp_fit(Xb,yb); out['GP fit, 20 BO (training)']=met(yb,g20.predict(u(Xb)))
    print('GP20 kernel',g20.kernel_, 'ext pred', np.round(g20.predict(u(Xe)),2))
    out['RSM neutral external, 8']=met(ye,rsm_e)
    out['GP neutral external, 8']=met(ye,gp_e)
    # pooled 58 repeated 5-fold CV
    Xa=np.vstack([Xc,Xb,Xe]); ya=np.concatenate([yc,yb,ye])
    terms=[(0,),(1,),(2,),(3,),(0,0),(1,1),(2,2),(3,3),(0,1),(1,3)]
    def des(X):
        Z=coded(X); return np.column_stack([np.prod(Z[:,list(t)],axis=1) for t in terms])
    yt_r,yp_r,yp_g=[],[],[]
    for tr,te in RepeatedKFold(n_splits=5,n_repeats=20,random_state=0).split(Xa):
        lr=LinearRegression().fit(des(Xa[tr]),ya[tr]); yp_r+=list(lr.predict(des(Xa[te])))
        yp_g+=list(gp_fit(Xa[tr],ya[tr]).predict(u(Xa[te]))); yt_r+=list(ya[te])
    out['RSM pooled 5x20 CV, 58']=met(yt_r,yp_r); out['GP pooled 5x20 CV, 58']=met(yt_r,yp_g)
    for k,v in out.items(): print(f"{k:32s} "+"  ".join(f"{a}={b:.3f}" for a,b in v.items()))
    json.dump({k:{a:float(b) for a,b in v.items()} for k,v in out.items()},open('results/metrics.json','w'),indent=1)
    np.save('results/gp_loo.npy',loo)
