import matplotlib.pyplot as plt
import os

# Ensure the simulation folder exists
SIM_FOLDER = "simulation"
os.makedirs(SIM_FOLDER, exist_ok=True)


def plotChain(chain_data, filename):
    steps = [step['step'] for step in chain_data]
    regular = [step['regular_result'] for step in chain_data]
    mapped = [step['mapped_result'] for step in chain_data]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, regular, 'o-', label='Regular float')
    plt.plot(steps, mapped, 's-', label='Mapped multiplication')
    plt.title(f"Chained Multiplication")
    plt.xlabel("Step")
    plt.ylabel("Value")
    # Fix axes so different runs are directly comparable
    plt.xlim(0, max(steps) if steps else 1)
    plt.ylim(-2.2, 2.2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    full_path = os.path.join(SIM_FOLDER, filename)
    plt.savefig(full_path)
    print(f"Plot saved as {full_path}")
    plt.close()
