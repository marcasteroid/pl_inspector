import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence

def plot_probability_bar_chart(
    probabilities: Sequence[float],
    class_names: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots a bar chart of class probabilities.

    Args:
        probabilities: A sequence of probability values.
        class_names: Optional sequence of class names corresponding to the probabilities.
        ax: Optional matplotlib Axes object to plot on. If None, a new plot is created.

    Returns:
        The matplotlib Axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    x_positions = np.arange(len(probabilities))
    ax.bar(x_positions, probabilities)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")

    if class_names is not None:
        if len(class_names) != len(probabilities):
            raise ValueError("Length of class_names must match length of probabilities.")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(class_names, rotation=45, ha="right")

    return ax

def plot_top_amplitude_magnitudes(
    amplitudes: Sequence[float],
    top_k: int = 10,
    feature_names: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots a bar chart of the top-k highest amplitude magnitudes.

    Args:
        amplitudes: A sequence of amplitude values.
        top_k: The number of top amplitudes to plot.
        feature_names: Optional sequence of names corresponding to the amplitudes.
        ax: Optional matplotlib Axes object to plot on. If None, a new plot is created.

    Returns:
        The matplotlib Axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    amplitudes_array = np.abs(np.array(amplitudes))
    # Ensure top_k does not exceed the number of available elements
    top_k = min(top_k, len(amplitudes_array))
    top_indices = np.argsort(amplitudes_array)[-top_k:][::-1]
    top_amplitudes = amplitudes_array[top_indices]

    x_positions = np.arange(len(top_amplitudes))
    ax.bar(x_positions, top_amplitudes)
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Top {top_k} Amplitude Magnitudes")

    if feature_names is not None:
        names = [feature_names[i] for i in top_indices]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(names, rotation=45, ha="right")
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(i) for i in top_indices])

    return ax

def plot_gradient_norm_history(
    grad_norms: Sequence[float],
    steps: Optional[Sequence[int]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots the history of gradient norms over time.

    Args:
        grad_norms: A sequence of gradient norm values.
        steps: Optional sequence of step numbers or epochs. If None, uses index.
        ax: Optional matplotlib Axes object to plot on. If None, a new plot is created.

    Returns:
        The matplotlib Axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if steps is None:
        steps = list(range(len(grad_norms)))

    ax.plot(steps, grad_norms, marker='o', markersize=3, linestyle='-')
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm History")
    ax.grid(True, linestyle='--', alpha=0.6)

    return ax

def plot_loss_curve(
    losses: Sequence[float],
    steps: Optional[Sequence[int]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots a loss curve over time.

    Args:
        losses: A sequence of loss values.
        steps: Optional sequence of step numbers or epochs. If None, uses index.
        ax: Optional matplotlib Axes object to plot on. If None, a new plot is created.

    Returns:
        The matplotlib Axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if steps is None:
        steps = list(range(len(losses)))

    ax.plot(steps, losses, linestyle='-', linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True, linestyle='--', alpha=0.6)

    return ax
