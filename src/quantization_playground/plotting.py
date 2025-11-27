from pathlib import Path
import matplotlib.pyplot as plt

from .paths import SIMULATION_DIR

# Ensure the simulation folder exists
Path(SIMULATION_DIR).mkdir(parents=True, exist_ok=True)


def plotChain(
    chain_data,
    filename,
    float_range=None,
    title=None,
    legend_labels=None,
    baseline_label=None,
    lookup_label=None,
    value_bits=None,
):
    steps = [step['step'] for step in chain_data]
    regular = [step['regular_result'] for step in chain_data]
    mapped = [step['mapped_result'] for step in chain_data]
    max_range = max(abs(float_range[0]), abs(float_range[1])) if float_range else 2.0
    margin = max_range * 0.1

    bits_label = f"{value_bits}-bit values" if value_bits else "uint8 values"
    labels = legend_labels or {
        "float": baseline_label or "Baseline (float16)",
        "mapped": lookup_label or f"Lookup table ({bits_label})",
    }
    plot_title = title or "Quantized vs float chain"

    plt.figure(figsize=(10, 5))
    plt.plot(steps, regular, 'o-', label=labels.get("float", "Baseline (float16)"))
    plt.plot(steps, mapped, 's-', label=labels.get("mapped", "Lookup table (uint8)"))
    plt.title(plot_title)
    plt.xlabel("Chain step")
    plt.ylabel("Accumulated product")
    # Fix axes so different runs are directly comparable
    plt.xlim(0, max(steps) if steps else 1)
    plt.ylim(-max_range - margin, max_range + margin)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    full_path = SIMULATION_DIR / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path)
    print(f"Plot saved as {full_path}")
    plt.close()
