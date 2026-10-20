import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
R={}
for f in ['results/res_bo.json','results/res_other.json']: R.update(json.load(open(f))['results'])
M=['GP_EI','PSO','GA','Random','LHS']; lab={'GP_EI':'GP/EI','PSO':'PSO','GA':'GA','Random':'Uniform\nrandom','LHS':'Single\nLHS'}
col={'20':'#9ecae1','30':'#3182bd'}
fig,(a,b)=plt.subplots(1,2,figsize=(10,4.2))
for k,B in enumerate(('20','30')):
    data=[[x['regret'] for x in R[m][B]] for m in M]; pos=np.arange(len(M))+(-0.18 if B=='20' else 0.18)
    bp=a.boxplot(data,positions=pos,widths=0.32,patch_artist=True,showfliers=False)
    for p in bp['boxes']: p.set_facecolor(col[B]); p.set_edgecolor('k')
    for med in bp['medians']: med.set_color('k')
    rng=np.random.default_rng(k)
    for i,dd in enumerate(data): a.plot(pos[i]+rng.uniform(-0.08,0.08,len(dd)),dd,'.',ms=2.5,color='k',alpha=0.35)
    a.plot([],[],'s',color=col[B],mec='k',label=f'{B} design points')
a.set_xticks(range(len(M))); a.set_xticklabels([lab[m] for m in M]); a.set_ylabel('Final regret (U/L)'); a.set_yscale('log'); a.set_ylim(0.03,30); a.legend(fontsize=8,frameon=False)
a.set_title('(a) Final regret on the fitted quadratic surface',loc='left',weight='bold',fontsize=9.5)
def wil(k,n):
    z=norm.ppf(.975); p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return c-h,c+h
for B in ('20','30'):
    x=np.arange(len(M))+(-0.18 if B=='20' else 0.18); ps=[];lo=[];hi=[]
    for m in M:
        k=sum(v['hit05'] for v in R[m][B]); p=k/60; l,h=wil(k,60); ps.append(100*p); lo.append(100*(p-l)); hi.append(100*(h-p))
    b.bar(x,ps,width=0.34,color=col[B],edgecolor='k',yerr=[lo,hi],capsize=3,label=f'by {B} design points')
b.set_xticks(range(len(M))); b.set_xticklabels([lab[m] for m in M]); b.set_ylabel('Campaigns reaching within 0.5 U/L\nof the surface maximum (%)'); b.set_ylim(0,105); b.legend(fontsize=8,frameon=False)
b.set_title('(b) Success proportion (95% Wilson CI)',loc='left',weight='bold',fontsize=9.5)
for ax in (a,b): ax.spines[['top','right']].set_visible(False)
fig.tight_layout(); fig.savefig('figures/FigS1_GA_PSO.png',dpi=300,facecolor='white')
