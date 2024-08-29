import argparse
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import pearsonr, skew
from transformers import RobertaTokenizer


def plot_prc(output_dir, predictions, labels):
    """
    Plots Precision-Recall Curve (PRC) and saves the figure.

    Args:
        output_dir (Path): Directory where the PRC plot will be saved.
        predictions (list or array): Model predictions, probabilities
                                   between 0 and 1.
        labels (list or array): True labels (binary) corresponding
                              to predictions.
    """
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    auprc = auc(recall, precision)
    p_pos = np.mean(labels)

    plt.figure(figsize=(10, 5))

    plt.plot(recall, precision, label="Model", zorder=1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.text(0.1, 0.25, f"Area = {auprc:.2f}", fontsize=11, ha="center")

    # No-skill classifier baseline
    plt.axhline(y=p_pos, color="r", linestyle="--", label="No Skill Baseline")
    plt.text(-0.1, p_pos, f"{p_pos:.2f}", color="black", va="center")

    # Find the threshold where recall is closest to 0.5
    idx_closest_recall = np.argmin(np.abs(recall - 0.5))
    print(f"threshold (recall=0.5): {thresholds[idx_closest_recall]}")
    print()

    # Mark score thresholds (steps of 0.1)
    max_pred = round(max(predictions), 1)
    min_pred = round(min(predictions), 1)
    for threshold in np.arange(min_pred + 0.2, max_pred + 0.1, 0.1):
        index = np.argmin(np.abs(thresholds - threshold))

        print(
            f"threshold: {round(threshold, 2)}, "
            f"precision: {round(precision[index], 3)}, "
            f"recall: {round(recall[index], 3)}"
        )

        if round(threshold, 1) == round(max_pred - 0.1, 1):
            plt.scatter(
                recall[index],
                precision[index],
                marker="o",
                color="black",
                label="Score threshold",
                zorder=2,
            )
        else:
            plt.scatter(
                recall[index], precision[index], marker="o",
                color="black", zorder=2
            )
        plt.text(
            recall[index] + 0.01,
            precision[index] + 0.03,
            f"{threshold:.1f}",
            fontsize=9,
            verticalalignment="top",
            zorder=2,
        )
    print()

    plt.legend(loc="best")

    plt.savefig(output_dir.joinpath("prc.png"), dpi=300)
    plt.show()


def plot_auprc(root):
    """
    Plots the AUPRC (Area Under the Precision-Recall Curve) for
    different epochs and checkpoints, and saves the figure.

    Args:
        root (Path): Directory containing JSON files with AUPRC results.
    """
    files = root.glob("*.json")
    auprcs = []
    labels = []

    # Regex to extract epoch and checkpoint from the filenames
    pattern = re.compile(r"train_eval_512_(\d+)_(\d+)\.json")
    files = [f for f in files if pattern.match(f.name)]

    # Extract epoch and checkpoint and sort files
    files_sorted = sorted(
        files,
        key=lambda f: (
            int(pattern.search(f.name).group(1)),
            int(pattern.search(f.name).group(2)),
        ),
    )

    for file in files_sorted:
        with file.open("r") as f:
            results = json.load(f)
            auprcs.append(results["auprc"])

            # Extract epoch and checkpoint for labeling
            match = pattern.search(file.name)
            epoch = match.group(1)
            checkpoint = match.group(2)
            labels.append(f"{epoch}.{checkpoint}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(auprcs)), auprcs, marker="o", linestyle="-", color="b")
    plt.xlabel("Epoch.Checkpoint")
    plt.ylabel("AUPRC")
    plt.xticks(range(len(auprcs)), labels, rotation=45)

    plt.savefig(root.joinpath("auprc.png"), dpi=300)
    plt.show()


def plot_histogram(output_dir, file_name, predictions, ymax, bins=100):
    """
    Plots a histogram of confidence scores and saves the figure.

    Args:
        output_dir (Path): Directory where the histogram plot will be saved.
        file_name (str): Name of the histogram file.
        predictions (list or array): List of confidence scores ranging
                                   from 0 to 1.
        ymax (int): Maximum value for the y-axis.
        bins (int): Number of bins for the histogram. Default is 100.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(predictions, bins=bins, range=(0, 1),
             edgecolor="black", alpha=0.7)
    plt.xlabel("Confidence Scores")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.ylim(0, ymax)

    plt.savefig(output_dir.joinpath(f"{file_name}.png"), dpi=300)

    plt.show()


def plot_boxplot(output_dir, positives, negatives):
    """
    Plots boxplots of the positive and negative scores to infer skewness
    and calculates the skewness for each.
    Reports quartiles and whisker bounds for both distributions.

    Args:
        output_dir (Path or str): Directory where the output file will be saved.
        positives (list or array): List of positive scores ranging from 0 to 1.
        negatives (list or array): List of negative scores ranging from 0 to 1.
    """

    data = [positives, negatives]
    labels = ["Positive", "Negative"]

    plt.figure(figsize=(12, 6))
    plt.boxplot(
        data,
        vert=False,
        patch_artist=True,
        labels=labels,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red"),
    )
    plt.xlabel("Scores")
    plt.grid(True)

    plt.savefig(output_dir.joinpath("boxplot.png"), dpi=300)
    plt.show()

    calculate_boxplot_stats("Positives", positives)
    calculate_boxplot_stats("Negatives", negatives)


def calculate_boxplot_stats(label, data):
    """
    Calculates and prints statistics for a boxplot including mean, quartiles,
    whisker bounds, and skewness.

    Args:
        label (str): Label for the data being analyzed.
        data (list or array): Data to analyze.
    """
    mean = np.mean(data)
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    lower_whisker = min(data)
    upper_whisker = max(data)

    print(f"*** {label} ***")
    print(f"Mean: {mean:.4f}")
    print(f"Q1: {q1:.4f}, Median: {median:.4f}, Q3: {q3:.4f}")
    print(f"Lower Whisker: {lower_whisker:.4f}, "
          f"Upper Whisker: {upper_whisker:.4f}")

    skewness_value = skew(data)
    print(f"Skewness: {skewness_value:.4f}")
    print()


def plot_distribution_percentage(output_dir, predictions):
    """
    Plots a layered bar chart of the distribution of confidence scores
    in a single bar.

    Args:
        output_dir (Path): Directory where the percentage distribution plot
                         will be saved.
        predictions (list or array): List of confidence scores ranging
                                   from 0 to 1.
    """
    bins = np.arange(0, 1.1, 0.1)
    counts, _ = np.histogram(predictions, bins=bins)

    total_count = len(predictions)
    percentages = (counts / total_count) * 100

    _, ax = plt.subplots(figsize=(10, 4))

    # Plot each bin as a separate layer in the stacked bar chart
    print("score distribution per range of 0.1 (%)")
    bottom = np.zeros_like(percentages)
    for i, percentage in enumerate(percentages):
        print(round(percentage, 3))
        ax.barh(
            "Scores",
            percentage,
            height=2.5,
            left=bottom[0],
            label=f"{bins[i]:.1f}-{bins[i+1]:.1f}",
            color=plt.cm.viridis((i + 3) / (len(bins) + 3)),
        )
        bottom += percentage

        # Determine the position for the label
        if percentage < 5:
            ax.text(
                bottom[0] + percentage,
                -1.5,
                f"{percentage:.1f}%",
                va="center",
                ha="center",
                fontsize=10,
                color="black",
            )
        else:
            ax.text(
                bottom[0] - percentage / 2,
                "Scores",
                f"{percentage:.1f}%",
                va="center",
                ha="center",
                fontsize=10,
                color="black",
            )

    ax2 = ax.twiny()
    ticks = np.arange(0, total_count + 5000, 5000)
    labels = [str(int(tick)) for tick in ticks]

    ax2.set_xticks((np.array(ticks) / total_count) * 100)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Number of Items")

    ax.set_xlabel("Percentage of Total Items (%)")
    ax.set_title("Distribution of Confidence Scores")

    ax.set_ylim(-2, 1.5)
    plt.savefig(
        output_dir.joinpath("score_distribution_percentage.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.tight_layout()
    plt.show()


def plot_separate_distributions(output_dir, input_file):
    """
    Plots histograms and boxplots of positive and negative predictions.

    Args:
        output_dir (Path): Directory where the plots will be saved.
        input_file (dict): Dictionary containing 'labels'
                         and 'predictions_mult' keys.
    """
    labeled_predictions = list(
        zip(input_file["labels"], input_file["predictions_mult"])
    )
    positives = [
        prediction for label, prediction
        in labeled_predictions if round(label) == 1
    ]
    negatives = [
        prediction for label, prediction
        in labeled_predictions if round(label) == 0
    ]

    plot_histogram(output_dir, "histogram_positives", positives, 110)
    plot_histogram(output_dir, "histogram_negatives", negatives, 430)
    plot_boxplot(output_dir, positives, negatives)


def sort_results(results_file, data_file):
    """
    Sorts results into positive and negative categories, tokenizes the text,
    and returns the sorted lists.

    Args:
        results_file (dict): Dictionary containing 'predictions_mult', 'labels',
                           and 'idxs' keys.
        data_file (dict): Dictionary with detailed information
                        for each data entry.

    Returns:
        positives (list of dict): Sorted list of positive predictions.
        negatives (list of dict): Sorted list of negative predictions.
    """
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    results = zip(
        results_file["predictions_mult"],
        results_file["labels"], results_file["idxs"]
    )

    positives, negatives = [], []
    issues = set()
    for result in results:
        entry = data_file[result[2]]
        issues.add(entry["issue"])

        code_input = tokenizer.tokenize(entry["code"])
        nl_input = tokenizer.tokenize(entry["nl_input"])

        data_point = {
            "prediction": result[0],
            "code": entry["code"],
            "nl_input": entry["nl_input"],
            "code_len": len(code_input),
            "nl_len": len(nl_input),
            "path": entry["path"],
        }

        if round(result[1]) == 1:
            positives.append(data_point)
        else:
            negatives.append(data_point)

    positives.sort(key=lambda x: x["prediction"])
    negatives.sort(key=lambda x: x["prediction"])
    print(f"N issues: {len(issues)}")
    print()

    return positives, negatives


def plot_scatter(output_dir, xvalues, yvalues, xlabel, ylabel, set_name):
    """
    Plot scatter plot with linear regression line and correlation statistics.

    Args:
        output_dir (Path or str): Directory to save the plot.
        xvalues (list or array): X-axis values.
        yvalues (list or array): Y-axis values.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        set_name (str): Identifier for the plot filename.
    """
    xvalues = np.array(xvalues).reshape(-1, 1)
    yvalues = np.array(yvalues)

    model = LinearRegression()
    model.fit(xvalues, yvalues)

    slope = model.coef_[0]
    intercept = model.intercept_

    plt.figure(figsize=(10, 5))
    plt.scatter(
        xvalues, yvalues, alpha=0.6, edgecolors="none",
        s=30, label="Data points"
    )

    # Plot the regression line
    x_range = np.linspace(xvalues.min(), xvalues.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    plt.plot(x_range, y_pred, color="red", label="Regression line")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

    # Calculate Pearson correlation coefficient and p-value
    r, p = pearsonr(xvalues.flatten(), yvalues)
    print(f"Pearson r ({xlabel}, {ylabel}): {r:.4f}, p = {p:.4f}")
    print(f"Linear Regression: Slope = {slope:.4f}, "
          f"Intercept = {intercept:.4f}")

    plt.savefig(
        output_dir.joinpath(f"scatter_{xlabel}_{ylabel}_{set_name}.png"),
        dpi=300
    )
    plt.show()


def create_scatterplots(output_dir, results, set_name):
    """
    Create scatterplots for predictions vs code_lens
    and predictions vs nl_lens.

    Args:
        output_dir (Path): Directory to save the scatter plots.
        results (list of dicts): List of data points containing 'prediction', 'code_len', and 'nl_len'.
        set_name (str): Identifier for the plot filenames.
    """
    predictions = [result["prediction"] for result in results]
    code_lens = [result["code_len"] for result in results]
    nl_lens = [result["nl_len"] for result in results]

    print(f"*** {set_name} ***")
    plot_scatter(
        output_dir, code_lens, predictions, "#Code tokens",
        "Prediction score", set_name
    )
    plot_scatter(
        output_dir,
        nl_lens,
        predictions,
        "#Natural language tokens",
        "Prediction score",
        set_name,
    )
    print()


def save_quartile_items(output_dir, results, set_name):
    """
    Save the three items around each quartile point of the predictions as
    a JSON file.

    Args:
        output_file (str): Filename to save the quartile items.
        results (list of dicts): List of data points sorted by prediction value.
    """
    n = len(results)
    q1_index = n // 4
    median_index = n // 2
    q3_index = 3 * n // 4

    quartile_items = {
        "lowest": results[:10],
        "q1": results[q1_index - 5: q1_index + 5],
        "median": results[median_index - 5: median_index + 5],
        "q3": results[q3_index - 5: q3_index + 5],
        "highest": results[-10:],
    }

    output_file = output_dir.joinpath(f"quartile_cases_{set_name}.json")
    with output_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(quartile_items))


def token_len_analysis(output_dir, results_file, data_file):
    """
    Analyzes token lengths of positive and negative predictions,
    creates scatterplots, and saves quartile items.

    Args:
        output_dir (Path): Directory to save the analysis results.
        results_file (dict): Dictionary with model evaluation results.
        data_file (dict): Dictionary with detailed data entries.
    """
    positives, negatives = sort_results(results_file, data_file)

    create_scatterplots(output_dir, positives, "pos")
    create_scatterplots(output_dir, negatives, "neg")

    save_quartile_items(output_dir, positives, "pos")
    save_quartile_items(output_dir, negatives, "neg")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        help="path to directory containing training evaluation files",
    )
    parser.add_argument(
        "--model_results",
        type=str,
        help="json file name containing model evaluation results",
    )
    parser.add_argument(
        "--data_eval",
        type=str,
        help="path to json file containing evaluation dataset",
    )

    args = parser.parse_args()

    root = Path(args.root)

    plot_auprc(root)

    input_path = root.joinpath(args.model_results)
    with input_path.open("r", encoding="utf-8") as f:
        input_file = json.load(f)

    data_path = Path(args.data_eval)
    with data_path.open("r", encoding="utf-8") as f:
        data_file = json.load(f)

    plot_prc(root, input_file["predictions_mult"], input_file["labels"])
    plot_separate_distributions(root, input_file)
    plot_distribution_percentage(root, input_file["predictions_mult"])
    token_len_analysis(root, input_file, data_file)


if __name__ == "__main__":
    main()
