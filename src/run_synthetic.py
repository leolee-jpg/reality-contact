"""
Sweep reflexivity alpha from 0 to 1 and plot the "scissors":
the surface link I(c_t ; x_{t+1}) persists while reality contact
I_t = I(c_t ; x_{t+1} | x_t, a_t) collapses as the loop closes.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from synthetic import generate
from mi_estimators import cmi_binning, mi_binning, learned_cmi

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    alphas = np.linspace(0, 1, 11)
    cmi, mi = [], []
    for a in alphas:
        d = generate(alpha=a, n=20000, seed=0)
        Z = np.column_stack([d["x_t"], d["a_t"]])
        cmi.append(cmi_binning(d["c_t"], d["x_next"], Z))
        mi.append(mi_binning(d["c_t"], d["x_next"]))
        print(f"alpha={a:.1f}  reality contact I_t={cmi[-1]:.3f}  surface link={mi[-1]:.3f}")

    print("\nlearned-estimator cross-check (reality contact):")
    learned = {}
    for a in [0.0, 0.5, 1.0]:
        d = generate(alpha=a, n=20000, seed=0)
        Z = np.column_stack([d["x_t"], d["a_t"]])
        learned[a] = learned_cmi(d["c_t"], d["x_next"], Z)
        print(f"  alpha={a:.1f}  learned I_t={learned[a]:.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(alphas, mi, "o-", color="#8a94a0", lw=2,
            label=r"surface link  $I(c_t; x_{t+1})$")
    ax.plot(alphas, cmi, "o-", color="#0f6e74", lw=2,
            label=r"reality contact  $I_t = I(c_t; x_{t+1} \mid x_t, a_t)$")
    ax.scatter(list(learned), list(learned.values()), marker="x", s=90,
               color="#1a2430", zorder=5, label="reality contact, learned estimator")
    ax.set_xlabel(r"reflexivity $\alpha$  (0 = signal tracks the world, 1 = pure self-reference)")
    ax.set_ylabel("information about the next outcome (nats)")
    ax.set_title("The surface link persists while reality contact collapses")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(OUT, "scissors.png")
    fig.savefig(path, dpi=130)
    print(f"\nsaved {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
