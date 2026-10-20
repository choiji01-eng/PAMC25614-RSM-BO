"""Supporting in-silico comparison of GP/EI Bayesian optimization, a real-coded
genetic algorithm (GA), particle swarm optimization (PSO), uniform random sampling
and a single Latin hypercube design on the fitted reduced quadratic RSM surface.
All factors are handled in coded units (-2..+2). Noise SD = 1.153 U/L (CCD pure error
SD of triplicate means)."""
import numpy as np, json, sys, warnings
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
warnings.filterwarnings("ignore")
LO, HI, D = -2.0, 2.0, 4
NOISE = 1.153
def f(X):
    X = np.atleast_2d(X); x1,x2,x3,x4 = X.T
    return (37.172 + 2.425*x1 + 2.129*x2 + 1.318*x3 + 2.230*x4
            - 2.562*x1**2 - 1.930*x2**2 - 1.313*x3**2 - 2.114*x4**2
            + 1.519*x1*x2 + 0.759*x2*x4)
# noise-free maximum
best = max((minimize(lambda x: -f(x)[0], x0, bounds=[(LO,HI)]*D) for x0 in np.random.default_rng(0).uniform(LO,HI,(20,D))), key=lambda r:-r.fun)
FMAX, XMAX = -best.fun, best.x
def obs(X, rng): return f(X) + rng.normal(0, NOISE, len(np.atleast_2d(X)))
def lhs(n, rng): return LO + (HI-LO)*qmc.LatinHypercube(d=D, seed=rng).random(n)
def to_u(X): return (X-LO)/(HI-LO)
def from_u(U): return LO+(HI-LO)*U
def fit_gp(X, y, rng):
    k = C(1.0,(1e-3,1e3))*Matern(length_scale=np.ones(D),length_scale_bounds=(1e-2,1e2),nu=2.5)+WhiteKernel(1e-2,(1e-6,1e1))
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=int(rng.integers(1e9)))
    gp.fit(to_u(X), y); return gp
def gp_argmax_mean(gp, rng, n=2000):
    U = rng.random((n,D)); m = gp.predict(U); starts = U[np.argsort(-m)[:5]]
    res = [minimize(lambda u: -gp.predict(u[None])[0], s, bounds=[(0,1)]*D) for s in starts]
    return from_u(min(res, key=lambda r:r.fun).x)
def ei_next(gp, fbest, rng, xi=0.01, n=3000):
    def ei(U):
        m, s = gp.predict(np.atleast_2d(U), return_std=True); s = np.maximum(s,1e-9)
        z = (m-fbest-xi)/s; return (m-fbest-xi)*norm.cdf(z)+s*norm.pdf(z)
    U = rng.random((n,D)); e = ei(U); starts = U[np.argsort(-e)[:5]]
    res = [minimize(lambda u: -ei(u)[0], s0, bounds=[(0,1)]*D) for s0 in starts]
    return from_u(min(res, key=lambda r:r.fun).x)
def summarize(X, yobs, rec):
    true = f(X); return dict(regret=float(FMAX-f(rec)[0]), true=true.tolist())
def run_bo(budget, rng, checkpoints=(20,30)):
    X = lhs(8, rng); y = obs(X, rng); recs = {}
    while len(y) < budget:
        gp = fit_gp(X, y, rng)
        if len(y) in checkpoints: recs[len(y)] = gp_argmax_mean(gp, rng)
        xn = ei_next(gp, y.max(), rng)
        X = np.vstack([X, xn]); y = np.append(y, obs(xn, rng))
    recs[len(y)] = gp_argmax_mean(fit_gp(X, y, rng), rng)
    return X, y, recs
def run_random(budget, rng):
    X = rng.uniform(LO,HI,(budget,D)); y = obs(X, rng); return X, y, gp_argmax_mean(fit_gp(X,y,rng), rng)
def run_lhs(budget, rng):
    X = lhs(budget, rng); y = obs(X, rng); return X, y, gp_argmax_mean(fit_gp(X,y,rng), rng)
def run_ga(budget, rng, lam=4, pc=0.9, eta_m=20, eta_c=15):
    """(mu+lambda) elitist real-coded GA: mu=8 (LHS init), lambda=4 offspring/generation,
    binary tournament, SBX crossover, polynomial mutation (pm=1/D)."""
    X = lhs(8, rng); y = obs(X, rng); P, Py = X.copy(), y.copy()
    while len(y) < budget:
        k = min(lam, budget-len(y)); kids = []
        while len(kids) < k:
            def tour():
                i,j = rng.integers(len(P),size=2); return P[i] if Py[i]>=Py[j] else P[j]
            a, b = tour().copy(), tour().copy()
            if rng.random() < pc:
                for d in range(D):
                    u = rng.random(); beta = (2*u)**(1/(eta_c+1)) if u<=0.5 else (1/(2*(1-u)))**(1/(eta_c+1))
                    a[d], b[d] = 0.5*((1+beta)*a[d]+(1-beta)*b[d]), 0.5*((1-beta)*a[d]+(1+beta)*b[d])
            for c in (a,b):
                for d in range(D):
                    if rng.random() < 1/D:
                        u = rng.random(); dl = (2*u)**(1/(eta_m+1))-1 if u<0.5 else 1-(2*(1-u))**(1/(eta_m+1))
                        c[d] += dl*(HI-LO)
                kids.append(np.clip(c,LO,HI))
        kids = np.array(kids[:k]); ky = obs(kids, rng)
        X = np.vstack([X,kids]); y = np.append(y,ky)
        allP = np.vstack([P,kids]); allY = np.append(Py,ky); idx = np.argsort(-allY)[:8]
        P, Py = allP[idx], allY[idx]
    return X, y, X[np.argmax(y)]
def run_pso(budget, rng, n=4, w=0.7298, c1=1.49618, c2=1.49618):
    """Global-best PSO, 4 particles initialised at the 4 best of an 8-point LHS."""
    X = lhs(8, rng); y = obs(X, rng); idx = np.argsort(-y)[:n]
    pos = X[idx].copy(); pb, pby = pos.copy(), y[idx].copy(); vel = np.zeros_like(pos)
    g = X[np.argmax(y)].copy(); gy = y.max(); vmax = 0.5*(HI-LO)
    while len(y) < budget:
        for i in range(n):
            if len(y) >= budget: break
            r1, r2 = rng.random(D), rng.random(D)
            vel[i] = np.clip(w*vel[i]+c1*r1*(pb[i]-pos[i])+c2*r2*(g-pos[i]), -vmax, vmax)
            pos[i] = np.clip(pos[i]+vel[i], LO, HI); yi = obs(pos[i], rng)[0]
            X = np.vstack([X,pos[i]]); y = np.append(y,yi)
            if yi > pby[i]: pb[i], pby[i] = pos[i].copy(), yi
            if yi > gy: g, gy = pos[i].copy(), yi
    return X, y, X[np.argmax(y)]
METHODS = dict(GP_EI=run_bo, GA=run_ga, PSO=run_pso, Random=run_random, LHS=run_lhs)
def recs_generic(fn, rng, gp_rule):
    """Random/LHS: separate designs of size 20 and 30 analysed with the GP recommendation rule.
    GA/PSO: one 30-evaluation run, truncated at 20; recommendation = best observed point."""
    if gp_rule:
        return {B: fn(B, rng) for B in (20,30)}
    X, y, _ = fn(30, rng)
    return {B:(X[:B], y[:B], X[:B][np.argmax(y[:B])]) for B in (20,30)}
if __name__ == "__main__":
    NC = 60; out = {}
    names = sys.argv[2].split(",") if len(sys.argv)>2 else list(METHODS)
    for name in names:
        fn = METHODS[name]
        res = {20:[],30:[]}
        for c in range(NC):
            rng = np.random.default_rng(1000+c)  # same seed per campaign index across methods
            if name == "GP_EI":
                X30, y30, recs = run_bo(30, rng); runs = {B:(X30[:B], y30[:B], recs[B]) for B in (20,30)}
            else:
                runs = recs_generic(fn, rng, gp_rule=name in ("Random","LHS"))
            for B in (20,30):
                X, y, rec = runs[B]; tv = f(X)
                first = np.where(tv >= FMAX-0.5)[0]; first1 = np.where(tv >= FMAX-1.0)[0]
                res[B].append(dict(regret=float(FMAX-f(rec)[0]), hit05=bool(len(first)>0),
                                   first05=int(first[0]+1) if len(first) else None,
                                   first10=int(first1[0]+1) if len(first1) else None))
        out[name] = res; 
        for B in (20,30):
            r = np.array([d['regret'] for d in res[B]]); h = np.mean([d['hit05'] for d in res[B]])
            print(f"{name:7s} B={B}: median regret {np.median(r):.2f} (IQR {np.percentile(r,25):.2f}-{np.percentile(r,75):.2f}); within 0.5: {100*h:.0f}%", flush=True)
    json.dump(dict(FMAX=FMAX, XMAX=XMAX.tolist(), results=out), open(sys.argv[1] if len(sys.argv)>1 else "sim_results.json","w"), indent=1)
