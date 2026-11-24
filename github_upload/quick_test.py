import time
import numpy as np

if __package__:
    from . import generator, attacks, detectors, metrics
else:
    import generator, attacks, detectors, metrics

t = time.time()
data = generator.generate(seed=0)                      
print("Data shape:", data["X"].shape)                  

att = attacks.inject(data, "S4", np.random.default_rng(1))   
print("Attack events:", len(att["events"]), "| attacked readings:", int(att["label"].sum()))

flags, rules, _ = detectors.run_all(att["X"], att["status"])  
for name, f in flags.items():                                  
    r = metrics.evaluate(f, att["label"], att["events"])
    print(f"{name:9s} DR={r['DR']:6.1f}%  FPR={r['FPR']:.2f}%  delay={r['delay']:.1f}")
print(f"Done in {time.time() - t:.1f} s")
