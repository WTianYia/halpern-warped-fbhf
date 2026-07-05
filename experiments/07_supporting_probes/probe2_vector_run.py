import numpy as np, time, probe_lib as P
d=np.load("ref.npz"); xstar=d["xstar"]; ystar=d["ystar"]; nx=np.linalg.norm(xstar)
zstar=(xstar,ystar); chi,eta=P.chi,P.eta; g=chi-eta   # 用最优固定步长
def perr(z): return np.linalg.norm(z[0]-xstar)/nx
def znorm(a): return np.sqrt(np.linalg.norm(a[0])**2+(a[1]**2).sum())
def zadd(a,b,s=1.0): return (a[0]+s*b[0], a[1]+s*b[1])
def zsub(a,b): return (a[0]-b[0], a[1]-b[1])
# 参考: 纯 FBHF 线搜索 的 fixed-budget primal error
def ls_curve(Kmax,th=0.9):
    z=(P.z0[0].copy(),P.z0[1].copy()); gg=chi; h=[]
    for it in range(Kmax):
        x,y=z; gf=P.gradf(x); b0,b1=P.Bop(x,y); gg*=1.2
        while True:
            ix=x-gg*(gf+b0); iy=y-gg*b1; xr=ix; yr=P.projb(iy); r0,r1=P.Bop(xr,yr)
            lhs=gg*np.sqrt(np.linalg.norm(b0-r0)**2+((b1-r1)**2).sum()); rhs=th*np.sqrt(np.linalg.norm(x-xr)**2+((y-yr)**2).sum())
            if lhs<=rhs or gg<=1e-6: break
            gg*=0.5
        z=(xr+gg*(b0-r0),yr+gg*(b1-r1)); h.append(perr(z))
    return np.array(h)
# Halpern-FBHF (+ 可和向量校正), u=0
def halpern_curve(Kmax,c=0.0,mode='none',p=0.5):
    z=(P.z0[0].copy(),P.z0[1].copy()); zp=z; h=[]; budget=0.0
    for k in range(Kmax):
        lam=1.0/(k+1)
        Sz=P.Sg(z,g,P.precompute(z))
        y=(( 1-lam)*Sz[0], (1-lam)*Sz[1])   # u=0 => + lam*u = 0
        rho=c/((k+1)**(1+p))
        if mode=='none' or c==0: dk=(0*y[0],0*y[1])
        elif mode=='clair':      # 指向真解 z*，预算 rho（capped 不越过）
            dirv=zsub(zstar,y); nn=znorm(dirv); s=min(rho,nn)/max(nn,1e-12); dk=(s*dirv[0],s*dirv[1])
        elif mode=='mom':        # 动量方向（不用 z*，近似可学）
            dirv=zsub(z,zp); nn=znorm(dirv); s=(rho/max(nn,1e-12)) if nn>1e-12 else 0; dk=(s*dirv[0],s*dirv[1])
        zp=z; z=(y[0]+dk[0], y[1]+dk[1]); budget+=min(rho, 1e9); h.append(perr(z))
    return np.array(h), sum(c/((k+1)**(1+p)) for k in range(Kmax))
Kmax=2000
ls=ls_curve(Kmax)
base,_=halpern_curve(Kmax,0.0)
print(f"总预算 Σρ_k 参考: ||z*||={znorm(zstar):.2f}")
print(f"{'策略':<34}{'err@300':>10}{'err@1000':>10}{'err@2000':>10}{'Σρ_k':>8}")
def row(name,h,B=None): print(f"{name:<32}{h[299]:>10.2e}{h[999]:>10.2e}{h[1999]:>10.2e}{('%.1f'%B) if B is not None else '':>8}")
row("线搜索FBHF(无锚,参考)",ls)
row("Halpern-FBHF 无校正",base)
for c in [0.5,2,8,32]:
    hc,B=halpern_curve(Kmax,c,'clair'); row(f"Halpern+千里眼校正 c={c}",hc,B)
for c in [2,8,32]:
    hm,B=halpern_curve(Kmax,c,'mom'); row(f"Halpern+动量校正(可学) c={c}",hm,B)
print("\n解读: 看'千里眼校正'(乐观上界)能否把 Halpern 基线拉到 >= 线搜索; 以及'动量校正'(可学)能否兑现。")
