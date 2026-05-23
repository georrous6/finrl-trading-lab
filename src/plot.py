#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tbparse import SummaryReader



#Tags to plot for each mode
TAGS = {
    "train": [
        "rollout/ep_rew_mean",
        "financial/final_value",
        "financial/max_drawdown",
        "financial/mean_episode_return",
        "financial/sharpe_ratio",
        "financial/total_return",
    ],
    "test": [
        "backtest/portfolio_value",
        "backtest/sharpe_ratio",
        "backtest/total_return",
    ],
}


def find_tb_runs(
    root_dir: Path,
    model_filter=None,
    policy_filter=None,
):
    """
    Expected layout:

    ROOT/
    └── model/
        └── policy/
            └── run_*/
                └── events.out.tfevents...
    """

    runs = []

    for model_dir in root_dir.iterdir():

        if not model_dir.is_dir():
            continue

        if (
            model_filter is not None
            and model_dir.name != model_filter
        ):
            continue

        for policy_dir in model_dir.iterdir():

            if not policy_dir.is_dir():
                continue

            if (
                policy_filter is not None
                and policy_dir.name != policy_filter
            ):
                continue

            for run_dir in policy_dir.glob("run_*"):

                if not run_dir.is_dir():
                    continue

                event_files = list(
                    run_dir.glob(
                        "events.out.tfevents*"
                    )
                )

                if len(event_files) == 0:
                    continue

                label = (
                    f"{model_dir.name}/"
                    f"{policy_dir.name}/"
                    f"{run_dir.name}"
                )

                runs.append(
                    {
                        "model": model_dir.name,
                        "policy": policy_dir.name,
                        "run": run_dir.name,
                        "label": label,
                        "path": run_dir,
                    }
                )

    return runs


def load_metric(run_dir, tag):

    reader = SummaryReader(run_dir)

    df = reader.scalars

    metric = df[
        df["tag"] == tag
    ]

    if metric.empty:
        return None

    return (
        metric["step"].values,
        metric["value"].values,
    )


def plot_tag(
    runs,
    tag,
    output_dir,
):

    plt.figure(
        figsize=(10, 6)
    )

    found = False

    for run in runs:

        data = load_metric(
            run["path"],
            tag,
        )

        if data is None:
            continue

        steps, values = data

        plt.plot(
            steps,
            values,
            label=run["label"],
        )

        found = True

    if not found:
        plt.close()
        print(
            f"Skipping {tag}: no data"
        )
        return

    plt.title(tag)

    plt.xlabel("Step")

    plt.ylabel(tag)

    plt.grid(True)

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()

    fname = (
        tag.replace("/", "_")
        + ".png"
    )

    out = output_dir / fname

    plt.savefig(
        out,
        dpi=300,
    )

    print(
        f"Saved {out}"
    )

    plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "root_dir",
        type=Path,
    )

    parser.add_argument(
        "--mode",
        default="train",
        choices=[
            "train",
            "test",
        ],
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--policy",
        default=None,
    )

    args = parser.parse_args()

    output_dir = Path("docs") / "figures"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = find_tb_runs(
        args.root_dir,
        model_filter=args.model,
        policy_filter=args.policy,
    )

    print(
        f"Found {len(runs)} runs"
    )

    tags = TAGS[
        args.mode
    ]

    for tag in tags:

        plot_tag(
            runs,
            tag,
            output_dir,
        )


if __name__ == "__main__":
    main()