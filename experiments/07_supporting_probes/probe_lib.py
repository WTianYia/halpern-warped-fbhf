import numpy as np
np.random.seed(0)
class Ctr:
    def __init__(s): s.C=0;s.B=0;s.P=0
ctr=Ctr()
N=64
img=np.zeros((N,N)); img[10:54,10:54]=0.3; img[20:44,20:44]=0.7
yy,xx=np.mgrid[0:N,0:N]; img[(xx-32)**2+(yy-32)**2<=49]=1.0; img[8:56,30:34]=0.5
def gpsf(sz=7,sig=1.5):
    a=np.arange(sz)-sz//2; g=np.exp(-(a[:,None]**2+a[None,:]**2)/(2*sig**2)); return g/g.sum()
psf=gpsf(); otf=np.zeros((N,N)); s=psf.shape[0]//2
otf[:7,:7]=psf; otf=np.roll(np.roll(otf,-s,0),-s,1); OTF=np.fft.fft2(otf)
def K(x): return np.real(np.fft.ifft2(np.fft.fft2(x)*OTF))
def Kt(x): return np.real(np.fft.ifft2(np.fft.fft2(x)*np.conj(OTF)))
b=K(img)+0.01*np.random.randn(N,N); mu=0.02
def D(x):
    gx=np.zeros_like(x);gy=np.zeros_like(x);gx[:,:-1]=x[:,1:]-x[:,:-1];gy[:-1,:]=x[1:,:]-x[:-1,:];return np.stack([gx,gy])
def Dt(p):
    gx,gy=p[0],p[1];dx=np.zeros_like(gx);dy=np.zeros_like(gy)
    dx[:,1:]=gx[:,1:]-gx[:,:-1];dx[:,0]=gx[:,0];dy[1:,:]=gy[1:,:]-gy[:-1,:];dy[0,:]=gy[0,:];return -(dx+dy)
def gradf(x): ctr.C+=1; return Kt(K(x)-b)
def Bop(x,y): ctr.B+=1; return Dt(y),-D(x)
def projb(y):
    ctr.P+=1; nrm=np.sqrt(y[0]**2+y[1]**2); f=np.minimum(1.0,mu/np.maximum(nrm,1e-12)); return np.stack([y[0]*f,y[1]*f])
def opnorm(op,it=80):
    v=np.random.randn(N,N)
    for _ in range(it): v=op(v); v/=np.linalg.norm(v)
    return np.sqrt(np.linalg.norm(op(v))/np.linalg.norm(v))
nK=opnorm(lambda v:Kt(K(v))); nD=opnorm(lambda v:Dt(D(v)))
beta=1.0/nK**2; L=nD; chi=4*beta/(1+np.sqrt(1+16*beta**2*L**2)); eta=0.05*chi
def precompute(z): x,y=z; gf=gradf(x); b0,b1=Bop(x,y); return gf,b0,b1
def Sg(z,g,pre):
    x,y=z; gf,b0,b1=pre; ix=x-g*(gf+b0); iy=y-g*b1; xr=ix; yr=projb(iy); r0,r1=Bop(xr,yr)
    return (xr+g*(b0-r0), yr+g*(b1-r1))
z0=(np.zeros((N,N)),np.zeros((2,N,N)))
