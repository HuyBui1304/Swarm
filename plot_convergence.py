import matplotlib.pyplot as plt
import numpy as np

datasets = [
    "Disjoint\ncircles",
    "Intersecting\ncircles",
    "Trefoil",
    "Cat",
    "Lion",
    "Horse",
    "Human",
    "Diabetes",
    "COVID-19",
]

pso_epochs = [11, 16, 5, 16, 10, 18, 7,  8,  6]
gwo_epochs = [3,   0, 5,  0,  0,  0, 0, 10,  6]

pso_times  = [2368.28, 3072.05, 46.51, 3385.20, 1265.08, 3418.02, 627.82, 37.55, 161.17]
gwo_times  = [ 480.72,  155.11, 53.55,  124.05,   76.55,  150.00,  51.06, 38.04, 147.36]

x = np.arange(len(datasets))

# ── Plot 1: Convergence Epochs ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, pso_epochs, marker='o', color='steelblue',  label='PSO Epochs')
ax.plot(x, gwo_epochs, marker='s', color='darkorange', label='GWO Epochs')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=9, rotation=30, ha='right')
ax.set_ylabel('Epochs')
ax.set_xlabel('Dataset')
ax.set_title('Convergence Epochs Comparison')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('convergence_epochs.png', dpi=150)
plt.savefig('convergence_epochs.eps', format='eps')
plt.show()
print("Saved: convergence_epochs.png / .eps")

# ── Plot 2: Convergence Time ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, pso_times, marker='o', color='steelblue',  label='PSO Time (s)')
ax.plot(x, gwo_times, marker='s', color='darkorange', label='GWO Time (s)')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=9, rotation=30, ha='right')
ax.set_ylabel('Time (s)')
ax.set_xlabel('Dataset')
ax.set_title('Convergence Time Comparison')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('convergence_time.png', dpi=150)
plt.savefig('convergence_time.eps', format='eps')
plt.show()
print("Saved: convergence_time.png / .eps")
