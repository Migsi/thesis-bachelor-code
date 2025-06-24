import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

metric_names = ["val_acc", "train_acc"]
results_dir = "results"
output_dir = "results_plots"


def map_metric_friendly_name(metric_name):
    """Map metric names to friendly names."""
    if metric_name == "val_acc":
        return "Validation Accuracy"
    elif metric_name == "train_acc":
        return "Training Accuracy"
    elif metric_name == "val_loss":
        return "Validation Loss"
    elif metric_name == "train_loss":
        return "Training Loss"
    else:
        return metric_name


def load_scalar(run_dir, tag):
    """Load scalar data from tensorboard logs."""
    ea = EventAccumulator(run_dir)
    ea.Reload()
    if tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    return None, None


def load_scalar_with_time(run_dir, tag):
    """Load scalar data with wall time from tensorboard logs."""
    ea = EventAccumulator(run_dir)
    ea.Reload()
    if tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        times = [e.wall_time for e in events]
        return steps, values, times
    return None, None, None


def aggregate_metric(exp_path, metric_name):
    """Aggregate metric across multiple versions/runs."""
    version_paths = glob(os.path.join(exp_path, "logs", "version_*"))
    all_values = []

    for version in version_paths:
        steps, values = load_scalar(version, metric_name)
        if steps and values:
            all_values.append(values)

    if not all_values:
        return None, None, None

    min_len = min(len(v) for v in all_values)
    trimmed = [v[:min_len] for v in all_values]
    arr = np.array(trimmed)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return list(range(min_len)), mean, std


def aggregate_metric_with_time(exp_path, metric_name):
    """Aggregate metric with time across multiple versions/runs."""
    version_paths = glob(os.path.join(exp_path, "logs", "version_*"))
    all_values = []
    all_times = []

    for version in version_paths:
        steps, values, times = load_scalar_with_time(version, metric_name)
        if steps and values and times:
            all_values.append(values)
            all_times.append(times)

    if not all_values:
        return None, None, None

    min_len = min(len(v) for v in all_values)
    trimmed_values = [v[:min_len] for v in all_values]
    trimmed_times = [t[:min_len] for t in all_times]

    # Convert to relative time (seconds from start)
    relative_times = []
    for times in trimmed_times:
        start_time = times[0]
        relative_times.append([(t - start_time) / 60 for t in times])  # Convert to minutes

    values_arr = np.array(trimmed_values)
    times_arr = np.array(relative_times)

    values_mean = values_arr.mean(axis=0)
    values_std = values_arr.std(axis=0)
    times_mean = times_arr.mean(axis=0)

    return times_mean, values_mean, values_std


def get_experiment_info(experiment_string):
    """
    Parse an experiment string with format: experiment_name_wsX_experiment_type

    Args:
        experiment_string (str): String in format "experiment_name_wsX_experiment_type"
                               where X is the world size number

    Returns:
        tuple: (experiment_name, world_size, experiment_type)
               world_size is returned as an integer

    Raises:
        ValueError: If string format is invalid or world size format is incorrect
    """
    parts = experiment_string.split('_')

    if len(parts) < 3:
        raise ValueError("String must have at least 3 parts separated by underscores")

    # Find the world size part (should match pattern "wsX")
    ws_index = None
    for i, part in enumerate(parts):
        if part.startswith('ws') and part[2:].isdigit():
            ws_index = i
            break

    if ws_index is None:
        raise ValueError("No valid world size found (expected format 'wsX' where X is a number)")

    # Extract experiment name (all parts before world size)
    experiment_name = '_'.join(parts[:ws_index])

    # Extract world size
    world_size = int(parts[ws_index][2:])  # Remove 'ws' prefix and convert to int

    # Extract experiment type (all parts after world size)
    experiment_type = '_'.join(parts[ws_index + 1:])

    if not experiment_name:
        raise ValueError("Experiment name cannot be empty")

    if not experiment_type:
        raise ValueError("Experiment type cannot be empty")

    return experiment_name, world_size, experiment_type


def create_plot_for_experiment(exp_path, plot_name, output_dir):
    """Create and save plot for a single experiment."""
    exp_name = os.path.basename(exp_path)
    base_name, world_size, suffix = get_experiment_info(exp_name)

    # Check if experiment has any data
    has_data = False
    for metric_name in metric_names:
        steps, mean, std = aggregate_metric(exp_path, metric_name)
        if steps is not None:
            has_data = True
            break

    if not has_data:
        print(f"  No data found for experiment: {plot_name}")
        return

    # Create plot
    plt.figure(figsize=(10, 6))
    color_map = plt.get_cmap("tab10")

    for metric_idx, metric_name in enumerate(metric_names):
        steps, mean, std = aggregate_metric(exp_path, metric_name)
        if steps is None:
            continue

        color = color_map(metric_idx)

        # Use different line styles for different metric types
        if 'acc' in metric_name:
            line_style = "-"  # Solid for accuracy
        else:  # loss metrics
            line_style = "--"  # Dashed for loss

        label = f"{map_metric_friendly_name(metric_name)}"

        plt.plot(steps, mean, line_style, label=label, color=color, linewidth=2)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Epoch", fontsize=14)

    # Set appropriate y-label based on metrics present
    if any('acc' in m for m in metric_names) and any('loss' in m for m in metric_names):
        plt.ylabel("Accuracy / Loss", fontsize=14)
    elif any('acc' in m for m in metric_names):
        plt.ylabel("Accuracy", fontsize=14)
    else:
        plt.ylabel("Loss", fontsize=14)

    plt.title(f"{base_name} - World Size {world_size} - {suffix.upper()}", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plot_filename = f"{plot_name}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {plot_filename}")


def create_time_plot_for_experiment(exp_path, plot_name, output_dir):
    """Create and save accuracy/time plot for a single experiment."""
    exp_name = os.path.basename(exp_path)
    base_name, world_size, suffix = get_experiment_info(exp_name)

    # Check if experiment has any data
    has_data = False
    for metric_name in metric_names:
        times, mean, std = aggregate_metric_with_time(exp_path, metric_name)
        if times is not None:
            has_data = True
            break

    if not has_data:
        print(f"  No time data found for experiment: {plot_name}")
        return

    # Create plot
    plt.figure(figsize=(10, 6))
    color_map = plt.get_cmap("tab10")

    for metric_idx, metric_name in enumerate(metric_names):
        times, mean, std = aggregate_metric_with_time(exp_path, metric_name)
        if times is None:
            continue

        color = color_map(metric_idx)

        # Use different line styles for different metric types
        if 'acc' in metric_name:
            line_style = "-"  # Solid for accuracy
        else:  # loss metrics
            line_style = "--"  # Dashed for loss

        label = f"{map_metric_friendly_name(metric_name)}"

        plt.plot(times, mean, line_style, label=label, color=color, linewidth=2)
        plt.fill_between(times, mean - std, mean + std, alpha=0.2, color=color)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Time (minutes)", fontsize=14)

    # Set appropriate y-label based on metrics present
    if any('acc' in m for m in metric_names) and any('loss' in m for m in metric_names):
        plt.ylabel("Accuracy / Loss", fontsize=14)
    elif any('acc' in m for m in metric_names):
        plt.ylabel("Accuracy", fontsize=14)
    else:
        plt.ylabel("Loss", fontsize=14)

    plt.title(f"{base_name} - World Size {world_size} - {suffix.upper()} (Time)", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plot_filename = f"{plot_name}_time.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved time plot: {plot_filename}")


def create_variant_comparison_time_plots(experiments_dict, suffix, run_name, output_dir):
    """Create time-based comparison plots for different variants within the same parallelization method."""
    if len(experiments_dict) < 2:
        return

    plt.figure(figsize=(14, 8))
    color_map = plt.get_cmap("tab10")

    # Calculate total number of lines we'll plot
    num_experiments = len(experiments_dict)
    num_metrics = len(metric_names)
    total_lines = num_experiments * num_metrics

    # Use a more diverse color map if we have many lines
    if total_lines > 10:
        color_map = plt.get_cmap("tab20")

    line_idx = 0
    has_data = False

    for exp_idx, (exp_name, exp_path) in enumerate(experiments_dict.items()):
        base_name, world_size, _ = get_experiment_info(exp_name)

        for metric_idx, metric_name in enumerate(metric_names):
            times, mean, std = aggregate_metric_with_time(exp_path, metric_name)
            if times is None:
                continue

            has_data = True

            # Assign unique color to each line
            color = color_map(line_idx % color_map.N)

            # Use different line styles for different metric types
            if 'acc' in metric_name:
                line_style = "-"  # Solid for accuracy
            else:  # loss metrics
                line_style = "--"  # Dashed for loss

            label = f"{map_metric_friendly_name(metric_name)} ({base_name}, World Size {world_size})"

            plt.plot(times, mean, line_style, label=label, color=color, linewidth=2)
            plt.fill_between(times, mean - std, mean + std, alpha=0.2, color=color)

            line_idx += 1

    if has_data:
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Time (minutes)", fontsize=14)

        # Set appropriate y-label based on metrics present
        if any('acc' in m for m in metric_names) and any('loss' in m for m in metric_names):
            plt.ylabel("Accuracy / Loss", fontsize=14)
        elif any('acc' in m for m in metric_names):
            plt.ylabel("Accuracy", fontsize=14)
        else:
            plt.ylabel("Loss", fontsize=14)

        plt.title(f"{suffix.upper()} Variants Comparison (Time)", fontsize=16)
        plt.legend(fontsize=10, ncol=2)  # Use 2 columns for legend to save space
        plt.grid(True)
        plt.tight_layout()

        # Create comparisons subdirectory
        comparison_dir = os.path.join(output_dir, "comparisons")
        os.makedirs(comparison_dir, exist_ok=True)

        # Save comparison plot
        plot_filename = f"{run_name}_{suffix}_variants_comparison_time.png"
        plot_path = os.path.join(comparison_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved {suffix.upper()} variants time comparison plot: {plot_filename}")


def create_variant_comparison_plots(experiments_dict, suffix, run_name, output_dir):
    """Create comparison plots for different variants within the same parallelization method."""
    if len(experiments_dict) < 2:
        return

    plt.figure(figsize=(14, 8))
    color_map = plt.get_cmap("tab10")

    # Calculate total number of lines we'll plot
    num_experiments = len(experiments_dict)
    num_metrics = len(metric_names)
    total_lines = num_experiments * num_metrics

    # Use a more diverse color map if we have many lines
    if total_lines > 10:
        color_map = plt.get_cmap("tab20")

    line_idx = 0
    has_data = False

    for exp_idx, (exp_name, exp_path) in enumerate(experiments_dict.items()):
        base_name, world_size, _ = get_experiment_info(exp_name)

        for metric_idx, metric_name in enumerate(metric_names):
            steps, mean, std = aggregate_metric(exp_path, metric_name)
            if steps is None:
                continue

            has_data = True

            # Assign unique color to each line
            color = color_map(line_idx % color_map.N)

            # Use different line styles for different metric types
            if 'acc' in metric_name:
                line_style = "-"  # Solid for accuracy
            else:  # loss metrics
                line_style = "--"  # Dashed for loss

            label = f"{map_metric_friendly_name(metric_name)} ({base_name}, World Size {world_size})"

            plt.plot(steps, mean, line_style, label=label, color=color, linewidth=2)
            plt.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

            line_idx += 1

    if has_data:
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Epoch", fontsize=14)

        # Set appropriate y-label based on metrics present
        if any('acc' in m for m in metric_names) and any('loss' in m for m in metric_names):
            plt.ylabel("Accuracy / Loss", fontsize=14)
        elif any('acc' in m for m in metric_names):
            plt.ylabel("Accuracy", fontsize=14)
        else:
            plt.ylabel("Loss", fontsize=14)

        plt.title(f"{suffix.upper()} Variants Comparison", fontsize=16)
        plt.legend(fontsize=10, ncol=2)  # Use 2 columns for legend to save space
        plt.grid(True)
        plt.tight_layout()

        # Create comparisons subdirectory
        comparison_dir = os.path.join(output_dir, "comparisons")
        os.makedirs(comparison_dir, exist_ok=True)

        # Save comparison plot
        plot_filename = f"{run_name}_{suffix}_variants_comparison.png"
        plot_path = os.path.join(comparison_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved {suffix.upper()} variants comparison plot: {plot_filename}")


def main():
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all run directories
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return

    run_dirs = [d for d in os.listdir(results_dir)
                if os.path.isdir(os.path.join(results_dir, d))]

    if not run_dirs:
        print(f"No run directories found in '{results_dir}'!")
        return

    print(f"Found runs: {run_dirs}")

    # Process each run
    for run_name in run_dirs:
        print(f"\nProcessing run: {run_name}")
        run_path = os.path.join(results_dir, run_name)

        # Find experiment directories within this run
        experiment_dirs = [d for d in os.listdir(run_path)
                           if os.path.isdir(os.path.join(run_path, d))]

        if not experiment_dirs:
            print(f"  No experiment directories found in run '{run_name}'!")
            continue

        # Group experiments by suffix for this run
        experiments_by_suffix = {'ddp': {}, 'fsdp': {}, 'unknown': {}}

        for exp_name in experiment_dirs:
            exp_path = os.path.join(run_path, exp_name)
            base_name, world_size, suffix = get_experiment_info(exp_name)
            experiments_by_suffix[suffix][exp_name] = exp_path

            # Create individual plots for this experiment
            plot_name = f"{run_name}_{exp_name}"
            create_plot_for_experiment(exp_path, plot_name, output_dir)
            create_time_plot_for_experiment(exp_path, plot_name, output_dir)

        # Create comparison plots for each suffix type (ddp variants, fsdp variants)
        for suffix in ['ddp', 'fsdp']:
            if len(experiments_by_suffix[suffix]) > 1:  # Only create comparison if we have multiple experiments
                create_variant_comparison_plots(experiments_by_suffix[suffix], suffix, run_name, output_dir)
                create_variant_comparison_time_plots(experiments_by_suffix[suffix], suffix, run_name, output_dir)

        print(f"  Found experiments in {run_name}:")
        for suffix, exps in experiments_by_suffix.items():
            if exps:
                print(f"    {suffix.upper()}: {list(exps.keys())}")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()