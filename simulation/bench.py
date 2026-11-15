"""Usage: python bench.py [nx ny nz]  (default 256 102 77). Run from simulation/ so `sstream` imports."""
import sys, time
import warp as wp
from sstream.lbm import LbmSolver, LbmConfig

n = tuple(int(v) for v in sys.argv[1:4]) or (256, 102, 77)
if len(n) != 3:
    n = (256, 102, 77)
s = LbmSolver(LbmConfig(n, tau=0.56, smagorinsky=0.12, periodic=(False, False, False), inlet_speed=0.06))
s.initialize(0.06, 0, 0)
s.run(20)
wp.synchronize()
steps = 200
t = time.perf_counter()
s.run(steps)
wp.synchronize()
dt = time.perf_counter() - t
print(f"{n} on {s.device}: {s.cell_count * steps / dt / 1e6:.1f} MLUPs ({dt / steps * 1e3:.2f} ms/step)")
