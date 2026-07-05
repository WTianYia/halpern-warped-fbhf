"""
Track B 线路2：warped 预条件 FBHF（学习对角 SPD 核 M_k）
============================================================
可证版：warped 预解 + M^{-1} 半前向；(QF-M) 已在 [11] 自包含推导+数值零违例。
网络学 per-pixel 对角预条件 (τ,s)=M^{-1} 对角块（τ:原始, s:对偶）。

两种模式：
  --mode bv   : 有界变差（可证）。M^{-1} = base + Σ 增量，‖增量‖≤η_k=c/(k+1)^1.1（可和）。
  --mode free : 每步自由 (τ_k,s_k)（不可证，加速上界，作对照）。
稳定夹紧（两模式都强制）：τ∈[τlo,τhi]、s∈[slo,shi] 且 τ·s·‖D‖² ≤ ρ<1（半前向稳定）。

用法（Colab）：
  python trackB_warped_train.py --selftest
  python trackB_warped_train.py --train --mode bv   --size 96 --K 40 --iters 3000 --ckpt wbv.pt
  python trackB_warped_train.py --eval  --mode bv   --size 96 --K_eval 400 --ckpt wbv.pt
"""
import argparse, math, time
import numpy as np, torch, torch.nn as nn
DEV='cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 算子（与 trackB_train.py 一致：全网格高斯 OTF + D 精确伴随）----------
def gauss_otf(H,W,sigma,device):
    B=sigma.shape[0]
    yy=torch.arange(H,device=device).view(1,1,H,1)-H//2; xx=torch.arange(W,device=device).view(1,1,1,W)-W//2
    g=torch.exp(-(xx**2+yy**2)/(2*sigma.view(B,1,1,1)**2)); g=g/g.sum(dim=(-2,-1),keepdim=True)
    return torch.fft.fft2(torch.fft.ifftshift(g,dim=(-2,-1)))
def Kf(x,otf):  return torch.real(torch.fft.ifft2(torch.fft.fft2(x)*otf))
def Ktf(x,otf): return torch.real(torch.fft.ifft2(torch.fft.fft2(x)*torch.conj(otf)))
def Dop(x):
    gx=torch.zeros_like(x); gy=torch.zeros_like(x)
    gx[...,:,:-1]=x[...,:,1:]-x[...,:,:-1]; gy[...,:-1,:]=x[...,1:,:]-x[...,:-1,:]
    return torch.cat([gx,gy],dim=1)
def Dtop(p):
    px,py=p[:,0:1],p[:,1:2]; ax=torch.zeros_like(px); ay=torch.zeros_like(py)
    ax[...,:,0]=-px[...,:,0]; ax[...,:,1:-1]=px[...,:,:-2]-px[...,:,1:-1]; ax[...,:,-1]=px[...,:,-2]
    ay[...,0,:]=-py[...,0,:]; ay[...,1:-1,:]=py[...,:-2,:]-py[...,1:-1,:]; ay[...,-1,:]=py[...,-2,:]
    return ax+ay
def projb(y,mu):
    n=torch.sqrt(y[:,0:1]**2+y[:,1:2]**2+1e-12); return y*torch.clamp(mu/n,max=1.0)

class TV:
    def __init__(s,b,otf,mu): s.b,s.otf,s.mu=b,otf,mu
    def obj(s,x):
        d=Dop(x); tv=torch.sqrt(d[:,0:1]**2+d[:,1:2]**2+1e-12).sum(dim=(1,2,3))
        return 0.5*((Kf(x,s.otf)-s.b)**2).sum(dim=(1,2,3))+s.mu*tv

def normD(H,W,device):
    v=torch.randn(1,1,H,W,device=device)
    for _ in range(60): v=Dtop(Dop(v)); v=v/torch.norm(v)
    return torch.sqrt(torch.norm(Dtop(Dop(v)))/torch.norm(v)).item()

# ---------- warped FBHF 一步（τ,s = M^{-1} 对角，per-pixel）----------
def warped_step(prob, x, y, tau, s):
    """tau:(B,1,H,W) 原始步; s:(B,1,H,W) 对偶步。返回 T=(Tx,Ty), 预解点(px,py)。"""
    gx = Ktf(Kf(x,prob.otf)-prob.b, prob.otf) + Dtop(y)   # (B+C) 原始块 = ∇f + D*y
    px = x - tau*gx
    py = projb(y + s*Dop(x), prob.mu)
    Tx = px + tau*Dtop(y-py)
    Ty = py - s*Dop(x-px)
    return (Tx,Ty),(px,py)

# ---------- 预条件网络：per-pixel (τ,s)，夹紧 + 稳定投影 ----------
class PrecNet(nn.Module):
    def __init__(s,ch=32):
        super().__init__()
        s.net=nn.Sequential(nn.Conv2d(7,ch,3,padding=1),nn.ReLU(),
                            nn.Conv2d(ch,ch,3,padding=1),nn.ReLU(),nn.Conv2d(ch,2,3,padding=1))
    def forward(s,feat): return s.net(feat)  # 2 通道 raw -> 映射为 (τ,s)

def clamp_ts(raw, L, tlo,thi,slo,shi, rho=0.9):
    """raw:(B,2,H,W) -> τ,s 夹到区间 + 稳定条件 τs‖D‖²≤rho（保 (QF-M)/半前向稳定）。"""
    tau = tlo+(thi-tlo)*torch.sigmoid(raw[:,0:1])
    s   = slo+(shi-slo)*torch.sigmoid(raw[:,1:2])
    s   = torch.minimum(s, rho/(tau*L**2+1e-9))         # 强制 τ·s·‖D‖² ≤ rho
    return tau, s

def features(prob,x,y,px,py,x_prev):
    return torch.cat([x, x-px, x-x_prev, y, y-py], dim=1)

# ---------- 数据（同 trackB_train.py）----------
def gen_batch(B,H,W,device,seed=None):
    g=torch.Generator(device='cpu')
    if seed is not None: g.manual_seed(seed)
    imgs=torch.zeros(B,1,H,W)
    for i in range(B):
        for _ in range(int(torch.randint(3,7,(1,),generator=g))):
            val=float(torch.rand(1,generator=g))*0.8+0.2
            if int(torch.randint(0,2,(1,),generator=g))==0:
                x0=int(torch.randint(0,W-8,(1,),generator=g)); y0=int(torch.randint(0,H-8,(1,),generator=g))
                w=int(torch.randint(6,W//2,(1,),generator=g)); h=int(torch.randint(6,H//2,(1,),generator=g))
                imgs[i,0,y0:min(H,y0+h),x0:min(W,x0+w)]=val
            else:
                cy=int(torch.randint(8,H-8,(1,),generator=g)); cx=int(torch.randint(8,W-8,(1,),generator=g)); r=int(torch.randint(4,H//4,(1,),generator=g))
                yy,xx=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij'); imgs[i,0][(xx-cx)**2+(yy-cy)**2<=r*r]=val
    imgs=imgs.to(device); sigma=(torch.rand(B,generator=g)*1.5+1.0).to(device); otf=gauss_otf(H,W,sigma,device)
    noise=(torch.rand(B,1,1,1,generator=g)*0.015+0.005).to(device); b=Kf(imgs,otf)+noise*torch.randn(B,1,H,W,device=device)
    return b,otf,0.02,imgs

# ---------- 展开 ----------
def unroll(prob,x0,y0,K,net=None,mode='bv',L=2.83,cfg=None,fixed=None):
    """fixed=(tau,s) 标量 -> 固定预条件基线。net!=None -> 学习。"""
    x,y=x0,y0; x_prev=x0; xs=[]
    tlo,thi,slo,shi=cfg['tlo'],cfg['thi'],cfg['slo'],cfg['shi']
    if net is not None and mode=='bv':
        tau=torch.full_like(x, cfg['tau0']); s=torch.full_like(x, cfg['s0'])  # base M^{-1}
    for k in range(K):
        # 先用当前 (τ,s) 求预解点以取特征
        if fixed is not None:
            tau=torch.full_like(x,fixed[0]); s=torch.full_like(x,fixed[1])
        elif net is not None:
            _,(px0,py0)=warped_step(prob,x,y, torch.full_like(x,cfg['tau0']), torch.full_like(x,cfg['s0']))
            raw=net(features(prob,x,y,px0,py0,x_prev)); dtau,dsig=clamp_ts(raw,L,tlo,thi,slo,shi,cfg['rho'])
            if mode=='free':
                tau,s=dtau,dsig
            else:  # bv：增量可和累加
                eta=cfg['c']/((k+1)**1.1)
                tau=torch.clamp(tau+torch.clamp(dtau-tau,-eta,eta), tlo,thi)
                s  =torch.clamp(s  +torch.clamp(dsig-s,-eta,eta), slo,shi)
                s  =torch.minimum(s, cfg['rho']/(tau*L**2+1e-9))
        else:
            tau=torch.full_like(x,cfg['tau0']); s=torch.full_like(x,cfg['s0'])
        T,(px,py)=warped_step(prob,x,y,tau,s); x_prev=x; x,y=T; xs.append(x)
    return xs

def selftest():
    H=W=48;B=2;b,otf,mu,cl=gen_batch(B,H,W,DEV,0);prob=TV(b,otf,mu);L=normD(H,W,DEV)
    x=torch.randn(B,1,H,W,device=DEV);z=torch.randn(B,1,H,W,device=DEV);p=torch.randn(B,2,H,W,device=DEV)
    print(f"[伴随] |<Kx,z>-<x,Ktz>|={abs(((Kf(x,otf)*z).sum()-(x*Ktf(z,otf)).sum()).item()):.2e}  |<Dx,p>-<x,Dtp>|={abs(((Dop(x)*p).sum()-(x*Dtop(p)).sum()).item()):.2e}")
    print(f"[常数] ‖D‖={L:.3f}  χ≈{4/(1+math.sqrt(1+16*L**2)):.4f}  阈值 1/χ≈{(1+math.sqrt(1+16*L**2))/4:.3f}")
    cfg=dict(tlo=0.1,thi=1.8,slo=0.02,shi=0.12,rho=0.9,tau0=0.3,s0=0.1,c=0.5)
    x0=b.clone();y0=torch.zeros(B,2,H,W,device=DEV)
    xs=unroll(prob,x0,y0,300,mode='bv',L=L,cfg=cfg,fixed=(0.3,0.1))
    print(f"[warped 收敛] obj {prob.obj(x0).mean():.3f} -> {prob.obj(xs[-1]).mean():.3f} (应显著下降)")

def train(a):
    H=W=a.size;L=normD(H,W,DEV);net=PrecNet().to(DEV);opt=torch.optim.Adam(net.parameters(),a.lr)
    cfg=dict(tlo=0.1,thi=1.8,slo=0.02,shi=0.12,rho=0.9,tau0=0.3,s0=0.1,c=a.c)
    vb,votf,vmu,_=gen_batch(a.batch,H,W,DEV,999);best=1e9
    for it in range(1,a.iters+1):
        b,otf,mu,_=gen_batch(a.batch,H,W,DEV);prob=TV(b,otf,mu)
        x0=b.clone();y0=torch.zeros_like(b).repeat(1,2,1,1)
        xs=unroll(prob,x0,y0,a.K,net=net,mode=a.mode,L=L,cfg=cfg)
        loss=sum((k+1)*prob.obj(xs[k]).mean() for k in range(a.K))/sum(range(1,a.K+1))
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
        if it%a.eval_every==0:
            net.eval()
            with torch.no_grad():
                pv=TV(vb,votf,vmu);vx0=vb.clone();vy0=torch.zeros_like(vb).repeat(1,2,1,1)
                vl=pv.obj(unroll(pv,vx0,vy0,a.K,net=net,mode=a.mode,L=L,cfg=cfg)[-1]).mean().item()
                bl=pv.obj(unroll(pv,vx0,vy0,a.K,L=L,cfg=cfg,fixed=(0.3,0.1))[-1]).mean().item()
            net.train();tag=""
            if vl<best and math.isfinite(vl): best=vl;torch.save(net.state_dict(),a.ckpt);tag=" *saved"
            print(f"it{it:5d} loss{loss.item():.3f} | val learned {vl:.3f} vs fixed-precond {bl:.3f} (gain {bl-vl:+.3f}){tag}")
    print(f"done best={best:.3f} -> {a.ckpt}")

@torch.no_grad()
def evaluate(a):
    H=W=a.size;L=normD(H,W,DEV);net=PrecNet().to(DEV);net.load_state_dict(torch.load(a.ckpt,map_location=DEV));net.eval()
    cfg=dict(tlo=0.1,thi=1.8,slo=0.02,shi=0.12,rho=0.9,tau0=0.3,s0=0.1,c=a.c)
    b,otf,mu,clean=gen_batch(a.ntest,H,W,DEV,2024);prob=TV(b,otf,mu)
    x0=b.clone();y0=torch.zeros_like(b).repeat(1,2,1,1)
    chi=4/(1+math.sqrt(1+16*L**2))
    xstar=unroll(prob,x0,y0,4000,L=L,cfg=cfg,fixed=(chi-0.05*chi,0.1))[-1]  # 参考解（裸FBHF长跑，近似）
    nx=torch.sqrt((xstar**2).sum(dim=(1,2,3)))
    def re(xs): return [(torch.sqrt(((xk-xstar)**2).sum(dim=(1,2,3)))/nx).median().item() for xk in xs]
    Ke=a.K_eval; res={}
    res['plain-FBHF']=unroll(prob,x0,y0,Ke,L=L,cfg=cfg,fixed=(chi-0.05*chi,0.10))
    res['fixed-precond']=unroll(prob,x0,y0,Ke,L=L,cfg=cfg,fixed=(1.5,0.07))
    res['learned-warped(ours)']=unroll(prob,x0,y0,Ke,net=net,mode=a.mode,L=L,cfg=cfg)
    print(f"{'method':<26}{'err@K/2':>10}{'err@K':>10}")
    for k,xs in res.items(): r=re(xs); print(f"{k:<26}{r[Ke//2]:>10.2e}{r[-1]:>10.2e}")
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        plt.figure(figsize=(7,4.3))
        for k,xs in res.items(): plt.semilogy(range(1,len(xs)+1),re(xs),lw=2,label=k)
        plt.xlabel('iteration');plt.ylabel('median rel. primal error');plt.legend();plt.grid(alpha=.3)
        plt.title(f'Track B warped ({a.mode})');plt.tight_layout();plt.savefig('trackB_warped_eval.png',dpi=120)
        print("saved trackB_warped_eval.png")
    except Exception as e: print("plot skip",e)
    print("\n判据: learned-warped 是否稳定优于 fixed-precond 与 plain。mode=bv 时收敛可证。")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for f in ['selftest','train','eval']: ap.add_argument('--'+f,action='store_true')
    ap.add_argument('--mode',default='bv',choices=['bv','free'])
    ap.add_argument('--size',type=int,default=96); ap.add_argument('--K',type=int,default=40)
    ap.add_argument('--K_eval',type=int,default=400); ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--iters',type=int,default=3000); ap.add_argument('--eval_every',type=int,default=100); ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--c',type=float,default=0.5); ap.add_argument('--ntest',type=int,default=8)
    ap.add_argument('--ckpt',default='wbv.pt'); a=ap.parse_args()
    print("device:",DEV,"mode:",a.mode)
    if a.selftest: selftest()
    if a.train: train(a)
    if a.eval: evaluate(a)
    if not(a.selftest or a.train or a.eval): print("用法: --selftest | --train | --eval  (--mode bv|free)")
