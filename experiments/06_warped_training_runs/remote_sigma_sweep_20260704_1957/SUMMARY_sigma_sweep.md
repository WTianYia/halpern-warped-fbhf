# Track B sigma sweep results

Date: 2026-07-04

Settings: size=96, K=40, batch=16, iters=3000, K_eval=400, ntest=8.

| sigma | plain err@K | linesearch err@K | inertial err@K | momentum err@K | learned err@K | learned PSNR@K | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.25 | 4.32e-02 | 4.07e-02 | 4.70e-02 | 5.86e-02 | **9.16e-03** | **28.43** | green |
| 0.50 | 4.17e-02 | 3.91e-02 | 4.55e-02 | 7.26e-02 | **6.79e-03** | **28.59** | green, best |
| 0.90 | 4.34e-02 | 4.10e-02 | 4.70e-02 | 8.63e-02 | **1.17e-02** | **28.43** | green |

Conclusion: learned-dev beats line-search FBHF and inertial FBHF in all three sigma settings. sigma=0.5 is the best of this sweep by err@K and PSNR@K.
