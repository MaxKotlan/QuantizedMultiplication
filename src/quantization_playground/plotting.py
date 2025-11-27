from pathlib import Path
import matplotlib.pyplot as plt

from .paths import SIMULATION_DIR

# Ensure the simulation folder exists
Path(SIMULATION_DIR).mkdir(parents=True, exist_ok=True)


def plotChain(chain_data, filename, float_range=None):
    steps = [step['step'] for step in chain_data]
    regular = [step['regular_result'] for step in chain_data]
    mapped = [step['mapped_result'] for step in chain_data]
    max_range = max(abs(float_range[0]), abs(float_range[1])) if float_range else 2.0
    margin = max_range * 0.1

    plt.figure(figsize=(10, 5))
    plt.plot(steps, regular, 'o-', label='Regular float')
    plt.plot(steps, mapped, 's-', label='Mapped multiplication')
    plt.title(f"Chained Multiplication")
    plt.xlabel("Step")
    plt.ylabel("Value")
    # Fix axes so different runs are directly comparable
    plt.xlim(0, max(steps) if steps else 1)
    plt.ylim(-max_range - margin, max_range + margin)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    full_path = SIMULATION_DIR / filename
    plt.savefig(full_path)
    print(f"Plot saved as {full_path}")
    plt.close()
