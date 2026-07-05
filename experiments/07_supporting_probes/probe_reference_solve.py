import numpy as np, time, probe_lib as P
t=time.time(); z=(P.z0[0].copy(),P.z0[1].copy()); g=0.5*P.chi
for _ in range(30000): z=P.Sg(z,g,P.precompute(z))
xstar=z[0]; nx=np.linalg.norm(xstar)
res=np.linalg.norm(P.Sg(z,g,P.precompute(z))[0]-xstar)/nx
np.savez("ref.npz",xstar=xstar,ystar=z[1])
print(f"||K||={P.nK:.4f} ||D||={P.nD:.4f} beta={P.beta:.3f} L={P.L:.3f} chi={P.chi:.5f} eta={P.eta:.5f}")
print(f"参考解就绪({time.time()-t:.1f}s) ||x*||={nx:.4f} 不动点残差={res:.2e}")
