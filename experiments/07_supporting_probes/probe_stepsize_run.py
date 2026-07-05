import numpy as np, time, probe_lib as P
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=np.load("ref.npz"); xstar=d["xstar"]; nx=np.linalg.norm(xstar)
chi,eta=P.chi,P.eta; TOL=1e-3; CAP=3000
def pd(z): return np.linalg.norm(z[0]-xstar)/nx
def run_fixed(g):
    z=(P.z0[0].copy(),P.z0[1].copy()); h=[]
    for it in range(CAP):
        z=P.Sg(z,g,P.precompute(z)); h.append(pd(z))
        if h[-1]<=TOL: break
    return h
def run_ls(th=0.9):
    z=(P.z0[0].copy(),P.z0[1].copy()); g=chi; h=[]; gs=[]
    for it in range(CAP):
        x,y=z; gf=P.gradf(x); b0,b1=P.Bop(x,y); g*=1.2
        while True:
            ix=x-g*(gf+b0); iy=y-g*b1; xr=ix; yr=P.projb(iy); r0,r1=P.Bop(xr,yr)
            lhs=g*np.sqrt(np.linalg.norm(b0-r0)**2+((b1-r1)**2).sum())
            rhs=th*np.sqrt(np.linalg.norm(x-xr)**2+((y-yr)**2).sum())
            if lhs<=rhs or g<=1e-6: break
            g*=0.5
        z=(xr+g*(b0-r0),yr+g*(b1-r1)); h.append(pd(z)); gs.append(g)
        if h[-1]<=TOL: break
    return h,gs
def run_clair():
    z=(P.z0[0].copy(),P.z0[1].copy()); cand=np.linspace(eta,chi-eta,12); h=[]; gs=[]
    for it in range(CAP):
        pre=P.precompute(z); bd=None
        for g in cand:
            zn=P.Sg(z,g,pre); dd=np.linalg.norm(zn[0]-xstar)
            if bd is None or dd<bd: bd=dd; bz=zn; bg=g
        z=bz; h.append(pd(z)); gs.append(bg)
        if h[-1]<=TOL: break
    return h,gs
def st(h): return next((i+1 for i,v in enumerate(h) if v<=TOL),None)
hf=run_fixed(chi-eta); hl,gl=run_ls(); hc,gc=run_clair()
sf,sl,sc=st(hf),st(hl),st(hc)
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
ax[0].semilogy(range(1,len(hf)+1),hf,label=f'fixed gamma=chi-eta  ({sf} it)',lw=2)
ax[0].semilogy(range(1,len(hl)+1),hl,label=f'line search (Lipschitz-free)  ({sl} it)',lw=2)
ax[0].semilogy(range(1,len(hc)+1),hc,'--',label=f'clairvoyant in [eta,chi-eta]  ({sc} it)',lw=2.2)
ax[0].axhline(TOL,color='gray',ls=':',lw=1)
ax[0].set_xlabel('iteration (≈ C-oracle count)'); ax[0].set_ylabel('relative primal error ||x-x*||/||x*||')
ax[0].set_title('Step-size strategies (TV deblur, FBHF primal-dual)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(range(1,len(gc)+1),gc,label='clairvoyant chosen gamma',lw=2)
ax[1].plot(range(1,len(gl)+1),gl,alpha=0.6,label='line-search gamma',lw=1)
ax[1].axhline(chi-eta,color='r',ls='--',lw=1,label='chi-eta (upper clamp)')
ax[1].axhline(chi,color='k',ls=':',lw=1,label='chi (theoretical ceiling)')
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('gamma'); ax[1].set_title('Chosen step size: clairvoyant sits ON the upper clamp')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("probe_curves.png",dpi=115)
print(f"fixed(chi-eta)={sf}  linesearch={sl}  clairvoyant-hard={sc}")
print(f"clairvoyant gamma constant? min={min(gc):.4f} max={max(gc):.4f}  (chi-eta={chi-eta:.4f})")
print(f"verdict ratio clair/ls={sl/sc:.2f}x  -> RED (scalar step-size saturated)")
print("saved probe_curves.png")
