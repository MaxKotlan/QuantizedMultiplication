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
    show_error=False,
    ylabel="Accumulated product",
):
    steps = [step['step'] for step in chain_data]
    regular = [step['regular_result'] for step in chain_data]
    mapped = [step['mapped_result'] for step in chain_data]
    errors = [step.get("percent_error", 0.0) for step in chain_data]
    max_range = max(abs(float_range[0]), abs(float_range[1])) if float_range else 2.0
    margin = max_range * 0.1

    bits_label = f"{value_bits}-bit values" if value_bits else "uint8 values"
    labels = legend_labels or {
        "float": baseline_label or "Baseline (float16)",
        "mapped": lookup_label or f"Int lookup ({bits_label})",
    }
    plot_title = title or "Quantized vs float chain"

    plt.figure(figsize=(10, 6 if show_error else 5))

    if show_error:
        gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
        ax_top = plt.subplot(gs[0])
        ax_err = plt.subplot(gs[1], sharex=ax_top)
    else:
        ax_top = plt.gca()
        ax_err = None

    ax_top.plot(steps, regular, 'o-', label=labels.get("float", "Baseline (float16)"))
    ax_top.plot(steps, mapped, 's-', label=labels.get("mapped", "Lookup table (uint8)"))
    ax_top.set_title(plot_title)
    ax_top.set_xlabel("Chain step")
    ax_top.set_ylabel(ylabel)
    ax_top.set_xlim(0, max(steps) if steps else 1)
    ax_top.set_ylim(-max_range - margin, max_range + margin)
    ax_top.legend()
    ax_top.grid(True, alpha=0.4)

    if show_error and ax_err is not None:
        ax_err.plot(steps, errors, 'r-', label="Percent error")
        ax_err.set_xlabel("Chain step")
        ax_err.set_ylabel("Percent error")
        ax_err.grid(True, alpha=0.4)
        ax_err.legend()

    plt.tight_layout()
    full_path = SIMULATION_DIR / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path)
    print(f"Plot saved as {full_path}")
    plt.close()
