"""
Track B 训练实验：FBHF + 学习型 ℓ² 下降配额 deviation（TV 图像去模糊，原始对偶）
=============================================================================
目标：验证"网络学 deviation 方向"能否在 fixed-budget 下打赢
      裸 FBHF / 线搜索 FBHF / 惯性 FBHF / 动量 deviation。

理论承重（务必保持）：
  迭代   z_{k+1} = S_γ(z_k) + d_k
  硬约束 ‖d_k‖² ≤ σ‖z_k − x_k‖²   （x_k = FBHF 预解点）
  => d_k 属 ℓ² 持久偏移，是 Track B 的安全入口。
  关键：d_k 由网络输出后**硬投影**进该球；手稿中仍需按 FBHF-deviation
       文献核对该投影对应的可吸收不等式，再声明弱收敛。

用法（Colab，自带 torch）：
  python trackB_train.py --selftest        # 先跑：算子伴随 + 裸FBHF收敛 自检
  python trackB_train.py --train           # 训练网络并保存 best.pt
  python trackB_train.py --eval            # 载入 best.pt，对标基线出表+图
依赖：torch, numpy, matplotlib（Colab 均自带）。GPU 自动使用，无 GPU 也能跑（慢些）。
"""
import argparse, math, time, os
import numpy as np
import torch, torch.nn as nn

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

# =========================================================================
# 1. 可微算子（与已验证的 numpy 台架一一对应）
# =========================================================================
def gauss_otf(H, W, sigma, device):
    """返回 (B,1,H,W) 复数 OTF；sigma: (B,) tensor。归一化高斯 => ||K||=1, beta=1。"""
    B = sigma.shape[0]
    yy = torch.arange(H, device=device).view(1,1,H,1) - H//2
    xx = torch.arange(W, device=device).view(1,1,1,W) - W//2
    s = sigma.view(B,1,1,1)
    g = torch.exp(-(xx**2 + yy**2) / (2*s**2))
    g = g / g.sum(dim=(-2,-1), keepdim=True)          # 归一化
    g = torch.fft.ifftshift(g, dim=(-2,-1))            # 中心移到原点
    return torch.fft.fft2(g)                           # (B,1,H,W) complex

def K_apply(x, otf):   return torch.real(torch.fft.ifft2(torch.fft.fft2(x)*otf))
def Kt_apply(x, otf):  return torch.real(torch.fft.ifft2(torch.fft.fft2(x)*torch.conj(otf)))

def Dop(x):
    """前向差分梯度 (B,1,H,W) -> (B,2,H,W)，Neumann 边界（末行/列差分置0）。"""
    gx = torch.zeros_like(x); gy = torch.zeros_like(x)
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return torch.cat([gx, gy], dim=1)

def Dtop(p):
    """D 的精确伴随 (B,2,H,W) -> (B,1,H,W)。使 B=(D*y,-Dx) 精确斜对称 => 精确单调。
    （已在 numpy 中验证 |<Dx,p>-<x,Dtp>|~1e-14、<Bz,z>~1e-14。）"""
    px = p[:, 0:1]; py = p[:, 1:2]
    ax = torch.zeros_like(px); ay = torch.zeros_like(py)
    ax[..., :, 0]    = -px[..., :, 0]
    ax[..., :, 1:-1] = px[..., :, :-2] - px[..., :, 1:-1]
    ax[..., :, -1]   = px[..., :, -2]
    ay[..., 0, :]    = -py[..., 0, :]
    ay[..., 1:-1, :] = py[..., :-2, :] - py[..., 1:-1, :]
    ay[..., -1, :]   = py[..., -2, :]
    return ax + ay

def proj_ball(y, mu):
    """逐像素把 2 通道对偶向量投到半径 mu 的 ℓ2 球。"""
    n = torch.sqrt(y[:, 0:1]**2 + y[:, 1:2]**2 + 1e-12)
    return y * torch.clamp(mu / n, max=1.0)

class TV:
    """一个 batch 的 TV 去模糊问题：min 0.5||Kx-b||^2 + mu||Dx||_{2,1}。"""
    def __init__(self, b, otf, mu, gamma):
        self.b, self.otf, self.mu, self.g = b, otf, mu, gamma
    def Sg(self, x, y):
        gf = Kt_apply(K_apply(x, self.otf) - self.b, self.otf)   # C 块（1 次 C）
        b0 = Dtop(y); b1 = -Dop(x)                                # Bz=(D*y,-Dx)
        ix = x - self.g*(gf + b0); iy = y - self.g*b1
        xr = ix; yr = proj_ball(iy, self.mu)                      # 预解
        r0 = Dtop(yr); r1 = -Dop(xr)                              # B(res)
        Sx = xr + self.g*(b0 - r0); Sy = yr + self.g*(b1 - r1)
        res = torch.sqrt(((x-xr)**2).sum(dim=(1,2,3)) + ((y-yr)**2).sum(dim=(1,2,3)) + 1e-12)  # (B,)
        return Sx, Sy, xr, yr, res                                # res=‖z−x_res‖ per sample
    def obj(self, x):
        d = Dop(x); tv = torch.sqrt(d[:,0:1]**2 + d[:,1:2]**2 + 1e-12).sum(dim=(1,2,3))
        fid = 0.5*((K_apply(x,self.otf)-self.b)**2).sum(dim=(1,2,3))
        return fid + self.mu*tv                                   # (B,)

# 步长常数：beta=1（归一化模糊），L=||D||（数值），chi 见 A20 Lemma 2.4(iii)
def compute_chi(H, W, device):
    v = torch.randn(1,1,H,W, device=device)
    for _ in range(60):
        v = Dtop(Dop(v)); v = v/torch.norm(v)
    L = torch.sqrt(torch.norm(Dtop(Dop(v)))/torch.norm(v)).item()
    beta = 1.0
    chi = 4*beta/(1+math.sqrt(1+16*beta**2*L**2))
    return chi, L

# =========================================================================
# 2. deviation 网络 + 硬投影
# =========================================================================
class DevNet(nn.Module):
    """输入 7 通道特征 -> 输出 3 通道原始 deviation (d_x:1, d_y:2)。~15K 参数。"""
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 3, 3, padding=1))
    def forward(self, feat): return self.net(feat)

def project_dev(q, res, sigma):
    """硬投影：‖(d_x,d_y)‖ ≤ sqrt(sigma)*res（per sample）。"""
    qx, qy = q[:, 0:1], q[:, 1:3]
    qnorm = torch.sqrt((qx**2).sum(dim=(1,2,3)) + (qy**2).sum(dim=(1,2,3)) + 1e-12)  # (B,)
    R = math.sqrt(sigma) * res                                                       # (B,)
    fac = torch.clamp(R/qnorm, max=1.0).view(-1,1,1,1)
    return qx*fac, qy*fac

def features(x, y, xr, yr, x_prev):
    return torch.cat([x, x-xr, x-x_prev, y, y-yr], dim=1)        # (B,7,H,W)

# =========================================================================
# 3. 数据生成（分片常值图 + 随机高斯模糊 + 噪声）
# =========================================================================
def gen_batch(B, H, W, device, seed=None):
    g = torch.Generator(device='cpu');
    if seed is not None: g.manual_seed(seed)
    imgs = torch.zeros(B,1,H,W)
    for i in range(B):
        n = int(torch.randint(3,7,(1,),generator=g))
        for _ in range(n):
            t = int(torch.randint(0,2,(1,),generator=g))
            val = float(torch.rand(1,generator=g))*0.8+0.2
            if t==0:  # 矩形
                x0=int(torch.randint(0,W-8,(1,),generator=g)); y0=int(torch.randint(0,H-8,(1,),generator=g))
                w=int(torch.randint(6,W//2,(1,),generator=g)); h=int(torch.randint(6,H//2,(1,),generator=g))
                imgs[i,0,y0:min(H,y0+h),x0:min(W,x0+w)]=val
            else:     # 圆
                cy=int(torch.randint(8,H-8,(1,),generator=g)); cx=int(torch.randint(8,W-8,(1,),generator=g))
                r=int(torch.randint(4,H//4,(1,),generator=g))
                yy,xx=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
                imgs[i,0][(xx-cx)**2+(yy-cy)**2<=r*r]=val
    imgs=imgs.to(device)
    sigma=(torch.rand(B,generator=g)*1.5+1.0).to(device)         # blur sigma ∈ [1,2.5]
    otf=gauss_otf(H,W,sigma,device)
    noise=(torch.rand(B,1,1,1,generator=g)*0.015+0.005).to(device)  # noise std ∈ [0.005,0.02]
    b=K_apply(imgs,otf)+noise*torch.randn(B,1,H,W,device=device)
    mu=0.02
    return b, otf, mu, imgs                                       # imgs=clean（仅评测用）

# =========================================================================
# 4. 展开求解器（可微；base=FBHF，可选注入 deviation）
# =========================================================================
def unroll(prob, x0, y0, K, net=None, sigma=0.5, mode='learn', alpha=0.0, gamma_in=None):
    """mode: 'plain' | 'inertial'(用 alpha,gamma_in) | 'momentum' | 'learn'(net)。
       返回 x 轨迹列表（no_grad 评测用）或最终 x（训练用，带图）。"""
    x, y = x0, y0; x_prev = x0; xs=[]
    for k in range(K):
        if mode=='inertial':
            w_x = x + alpha*(x - x_prev); w_y = y + alpha*(y - y_prev if k>0 else 0*y)
            p2 = TV(prob.b, prob.otf, prob.mu, gamma_in)
            Sx,Sy,xr,yr,res = p2.Sg(w_x, w_y)
            y_prev = y; x_prev = x; x, y = Sx, Sy; xs.append(x); continue
        Sx,Sy,xr,yr,res = prob.Sg(x, y)
        if mode=='plain':
            dx=0.0; dy=0.0
        elif mode=='momentum':
            mx=x-x_prev; my=y-(y_prev if k>0 else 0*y)
            qn=torch.sqrt((mx**2).sum(dim=(1,2,3))+(my**2).sum(dim=(1,2,3))+1e-12)
            R=math.sqrt(sigma)*res; fac=torch.clamp(R/qn,max=1.0).view(-1,1,1,1)
            dx=mx*fac; dy=my*fac
        else:  # learn
            q = net(features(x,y,xr,yr,x_prev)); dx,dy = project_dev(q,res,sigma)
        y_prev = y; x_prev = x
        x = Sx + dx; y = Sy + dy; xs.append(x)
    return xs

# 线搜索 FBHF（免-Lipschitz），单独实现（评测基线）
@torch.no_grad()
def unroll_ls(prob, x0, y0, K, theta=0.9, chi=None):
    x,y=x0,y0; g=chi; xs=[]
    for _ in range(K):
        gf=Kt_apply(K_apply(x,prob.otf)-prob.b,prob.otf); b0=Dtop(y); b1=-Dop(x); g=g*1.2
        for _ in range(40):
            ix=x-g*(gf+b0); iy=y-g*b1; xr=ix; yr=proj_ball(iy,prob.mu); r0=Dtop(yr); r1=-Dop(xr)
            lhs=g*torch.sqrt(((b0-r0)**2).sum(dim=(1,2,3))+((b1-r1)**2).sum(dim=(1,2,3))+1e-12)
            rhs=theta*torch.sqrt(((x-xr)**2).sum(dim=(1,2,3))+((y-yr)**2).sum(dim=(1,2,3))+1e-12)
            if bool((lhs<=rhs).all()) or g<=1e-6: break
            g=g*0.5
        x=xr+g*(b0-r0); y=yr+g*(b1-r1); xs.append(x)
    return xs

# =========================================================================
# 5. 自检：算子伴随 + 裸 FBHF 收敛
# =========================================================================
def selftest():
    H=W=48; B=2
    b,otf,mu,clean=gen_batch(B,H,W,DEV,seed=0)
    # 伴随检验 <Kx,z>=<x,Ktz>, <Dx,p>=<x,Dtp>
    x=torch.randn(B,1,H,W,device=DEV); z=torch.randn(B,1,H,W,device=DEV); p=torch.randn(B,2,H,W,device=DEV)
    a1=(K_apply(x,otf)*z).sum(); a2=(x*Kt_apply(z,otf)).sum()
    d1=(Dop(x)*p).sum(); d2=(x*Dtop(p)).sum()
    print(f"[伴随] |<Kx,z>-<x,Ktz>|={abs((a1-a2).item()):.2e}  |<Dx,p>-<x,Dtp>|={abs((d1-d2).item()):.2e}")
    chi,L=compute_chi(H,W,DEV); print(f"[常数] ||D||=L={L:.4f}  chi={chi:.5f}")
    prob=TV(b,otf,mu,chi-0.05*chi)
    x0=b.clone(); y0=torch.zeros(B,2,H,W,device=DEV)
    xs=unroll(prob,x0,y0,K=800,mode='plain')
    f0=prob.obj(x0).mean().item(); fK=prob.obj(xs[-1]).mean().item()
    print(f"[裸FBHF收敛] obj: {f0:.3f} -> {fK:.3f}  (应显著下降)")
    print("自检完成：伴随误差应 ~1e-4 以下，obj 应明显下降。")

# =========================================================================
# 6. 训练
# =========================================================================
def train(args):
    H=W=args.size; chi,L=compute_chi(H,W,DEV); gamma=chi-0.05*chi
    net=DevNet().to(DEV); opt=torch.optim.Adam(net.parameters(), lr=args.lr)
    # 固定验证集
    vb,votf,vmu,vclean=gen_batch(args.batch,H,W,DEV,seed=999)
    best=1e9
    for it in range(1,args.iters+1):
        b,otf,mu,clean=gen_batch(args.batch,H,W,DEV,seed=None)
        prob=TV(b,otf,mu,gamma); x0=b.clone(); y0=torch.zeros_like(b).repeat(1,2,1,1)
        x,y=x0,y0; x_prev=x0; loss=0.0; wsum=0.0
        for k in range(args.K):
            Sx,Sy,xr,yr,res=prob.Sg(x,y)
            q=net(features(x,y,xr,yr,x_prev)); dx,dy=project_dev(q,res,args.sigma)
            x_prev=x; x=Sx+dx; y=Sy+dy
            w=(k+1)                                    # 越晚权重越大（偏向 fixed-budget 末端）
            loss=loss+w*prob.obj(x).mean(); wsum+=w
        loss=loss/wsum
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        if it%args.eval_every==0:
            net.eval()
            with torch.no_grad():
                pv=TV(vb,votf,vmu,gamma); x0=vb.clone(); y0=torch.zeros_like(vb).repeat(1,2,1,1)
                xs=unroll(pv,x0,y0,args.K,net=net,sigma=args.sigma,mode='learn')
                vloss=pv.obj(xs[-1]).mean().item()
                xs0=unroll(pv,x0,y0,args.K,mode='plain'); base=pv.obj(xs0[-1]).mean().item()
            net.train()
            tag=""
            if vloss<best: best=vloss; torch.save(net.state_dict(), args.ckpt); tag=" *saved"
            print(f"it{it:5d} train{loss.item():.3f} | val learned {vloss:.3f} vs plain {base:.3f} "
                  f"(gain {base-vloss:+.3f}){tag}")
    print(f"训练完成，best val obj={best:.3f}，权重存 {args.ckpt}")

# =========================================================================
# 7. 评测：learned vs plain / linesearch / inertial / momentum
# =========================================================================
@torch.no_grad()
def evaluate(args):
    H=W=args.size; chi,L=compute_chi(H,W,DEV); gamma=chi-0.05*chi
    net=DevNet().to(DEV); net.load_state_dict(torch.load(args.ckpt,map_location=DEV)); net.eval()
    b,otf,mu,clean=gen_batch(args.ntest,H,W,DEV,seed=2024)     # held-out 测试集
    prob=TV(b,otf,mu,gamma); x0=b.clone(); y0=torch.zeros_like(b).repeat(1,2,1,1)
    # 参考解 x*（长跑）
    xs_ref=unroll(prob,x0,y0,4000,mode='plain'); xstar=xs_ref[-1]
    nx=torch.sqrt((xstar**2).sum(dim=(1,2,3)))
    Ke=args.K_eval
    curves={}
    curves['plain']=unroll(prob,x0,y0,Ke,mode='plain')
    curves['linesearch']=unroll_ls(prob,x0,y0,Ke,chi=chi)
    # 惯性最优 alpha/gamma（小网格）
    besti=None
    for gin in [0.4*chi,0.55*chi,0.7*chi]:
        for al in [0.2,0.35,0.5]:
            try: xi=unroll(prob,x0,y0,Ke,mode='inertial',alpha=al,gamma_in=gin)
            except Exception: continue
            if torch.isfinite(xi[-1]).all():
                e=(torch.sqrt(((xi[-1]-xstar)**2).sum(dim=(1,2,3)))/nx).mean().item()
                if besti is None or e<besti[0]: besti=(e,al,gin,xi)
    curves[f'inertial(α={besti[1]},γ={besti[2]:.3f})']=besti[3]
    curves['momentum-dev']=unroll(prob,x0,y0,Ke,mode='momentum',sigma=args.sigma)
    curves['learned-dev(ours)']=unroll(prob,x0,y0,Ke,net=net,sigma=args.sigma,mode='learn')
    # 指标：相对原始误差（中位数）随迭代
    def relerr(xs): return [ (torch.sqrt(((xk-xstar)**2).sum(dim=(1,2,3)))/nx).median().item() for xk in xs]
    def psnr(xs):   return [ (10*torch.log10(1.0/((xk-clean)**2).mean(dim=(1,2,3)))).median().item() for xk in xs]
    print(f"\n{'方法':<26}{'err@K/2':>10}{'err@K':>10}{'PSNR@K':>9}")
    for name,xs in curves.items():
        re=relerr(xs); ps=psnr(xs)
        print(f"{name:<24}{re[Ke//2]:>10.2e}{re[-1]:>10.2e}{ps[-1]:>9.2f}")
    # 出图
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        plt.figure(figsize=(7.5,4.5))
        for name,xs in curves.items():
            plt.semilogy(range(1,len(xs)+1), relerr(xs), label=name, lw=2)
        plt.xlabel('iteration'); plt.ylabel('median relative primal error'); plt.legend(fontsize=8)
        plt.title('Track B: learned deviation vs baselines'); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig('trackB_eval.png',dpi=120); print("图存 trackB_eval.png")
    except Exception as e: print("绘图跳过:",e)
    print("\n判据: 'learned-dev' 的 err@K / PSNR@K 是否稳定优于 linesearch 与 inertial。")

# =========================================================================
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--train',action='store_true')
    ap.add_argument('--eval',action='store_true')
    ap.add_argument('--size',type=int,default=96)      # 图像 H=W
    ap.add_argument('--K',type=int,default=40)          # 训练展开步数
    ap.add_argument('--K_eval',type=int,default=400)    # 评测步数
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--iters',type=int,default=3000)
    ap.add_argument('--eval_every',type=int,default=100)
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--sigma',type=float,default=0.5)   # deviation 预算 ‖d‖²≤σ‖z-x‖²
    ap.add_argument('--ntest',type=int,default=8)
    ap.add_argument('--ckpt',type=str,default='best.pt')
    args=ap.parse_args()
    print("device:",DEV)
    if args.selftest: selftest()
    if args.train:    train(args)
    if args.eval:     evaluate(args)
    if not (args.selftest or args.train or args.eval):
        print("用法: --selftest | --train | --eval （见文件头注释）")
