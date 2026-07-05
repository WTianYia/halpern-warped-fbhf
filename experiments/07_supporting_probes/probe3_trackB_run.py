import numpy as np, probe_lib as P
np.seterr(all='ignore')
d=np.load("ref.npz"); xstar=d["xstar"]; ystar=d["ystar"]; nx=np.linalg.norm(xstar); zstar=(xstar,ystar)
chi,eta=P.chi,P.eta; g=chi-eta; K=2600
def pe(z): return np.linalg.norm(z[0]-xstar)/nx
def zn(a): return np.sqrt(np.linalg.norm(a[0])**2+(a[1]**2).sum())
def fin(z): return np.isfinite(z[0]).all() and np.isfinite(z[1]).all()
def Sg(z,gam,pre):
    x,y=z; gf,b0,b1=pre; xr=x-gam*(gf+b0); yr=P.projb(y-gam*b1); r0,r1=P.Bop(xr,yr)
    return (xr+gam*(b0-r0), yr+gam*(b1-r1)), np.sqrt(np.linalg.norm(x-xr)**2+((y-yr)**2).sum())
def st(h,tol=1e-3):
    a=np.array(h); w=np.where(a<=tol)[0]; return int(w[0]+1) if len(w) else None
def plain():
    z=(P.z0[0].copy(),P.z0[1].copy()); h=[]
    for k in range(K): z,_=Sg(z,g,P.precompute(z)); h.append(pe(z))
    return h
def ls(th=0.9):
    z=(P.z0[0].copy(),P.z0[1].copy()); gg=chi; h=[]
    for it in range(K):
        x,y=z; gf=P.gradf(x); b0,b1=P.Bop(x,y); gg*=1.2
        while True:
            ix=x-gg*(gf+b0); iy=y-gg*b1; xr=ix; yr=P.projb(iy); r0,r1=P.Bop(xr,yr)
            if gg*np.sqrt(np.linalg.norm(b0-r0)**2+((b1-r1)**2).sum())<=th*np.sqrt(np.linalg.norm(x-xr)**2+((y-yr)**2).sum()) or gg<=1e-6: break
            gg*=0.5
        z=(xr+gg*(b0-r0),yr+gg*(b1-r1)); h.append(pe(z))
    return h
def inert(al,gin):
    z=(P.z0[0].copy(),P.z0[1].copy()); zp=z; h=[]
    for k in range(K):
        w=(z[0]+al*(z[0]-zp[0]),z[1]+al*(z[1]-zp[1])); S,_=Sg(w,gin,P.precompute(w)); zp=z; z=S
        if not fin(z): return None
        h.append(pe(z))
    return h
def dev(sig,mode):
    z=(P.z0[0].copy(),P.z0[1].copy()); zp=z; h=[]; R=np.sqrt(sig)
    for k in range(K):
        S,res=Sg(z,g,P.precompute(z)); cap=R*res
        dv=(zstar[0]-S[0],zstar[1]-S[1]) if mode=='clair' else (z[0]-zp[0],z[1]-zp[1])
        n=zn(dv); s=(min(cap,n)/n) if n>1e-12 else 0.0; zp=z; z=(S[0]+s*dv[0],S[1]+s*dv[1])
        if not fin(z): return None
        h.append(pe(z))
    return h
res={}
res['裸FBHF(γ=χ-η)']=plain(); res['线搜索FBHF']=ls()
best=None
for gin in [0.4*chi,0.55*chi,0.7*chi]:
    for al in [0.2,0.35,0.5]:
        hh=inert(al,gin)
        if hh is None: continue
        key=(st(hh) or 10**9, hh[-1])
        if best is None or key<best[0]: best=(key,al,gin,hh)
res[f'惯性FBHF(α={best[1]:.2f},γ={best[2]:.3f})']=best[3]
res['千里眼dev σ=0.5']=dev(0.5,'clair'); res['千里眼dev σ=0.9']=dev(0.9,'clair'); res['动量dev σ=0.9(可学)']=dev(0.9,'mom')
print(f"{'策略':<30}{'@300':>9}{'@1000':>9}{'@2000':>9}{'→1e-3':>7}")
for k,h in res.items(): print(f"{k:<28}{h[299]:>9.2e}{h[999]:>9.2e}{h[1999]:>9.2e}{str(st(h)):>7}")
inek=[k for k in res if k.startswith('惯性')][0]
a,b,c=st(res['线搜索FBHF']),st(res[inek]),st(res['千里眼dev σ=0.9'])
print(f"\n达标步 线搜索={a} 惯性={b} 千里眼dev(B天花板)={c}")
if b and c: print(f"千里眼dev/惯性={b/c:.2f}x ; /线搜索={a/c:.2f}x")
np.save("trackB_curves.npy",{k:np.array(v) for k,v in res.items()},allow_pickle=True)
print("saved")
