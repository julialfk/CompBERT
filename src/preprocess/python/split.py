import argparse
import random
import json
from pathlib import Path


def split(project_file: str, duplicate_file: str, data_file: str, output_dir: str, seed=42, split=0.8):
    random.seed(seed)

    project_file = Path(project_file)
    # Get the list of issue IDs
    with project_file.open("r") as f:
        issues = json.load(f)
        issues_keys = issues.keys()

    if duplicate_file is not None:
        duplicate_file = Path(duplicate_file)
        # Get the list of issue IDs
        with duplicate_file.open("r") as f:
            duplicate_commits = set(json.load(f))

        # Do not include issues that contain commits that were found in other issues.
        issues_keys = set(filter(lambda issue:
                                 all([commit not in duplicate_commits
                                     for commit in issues[issue]["commits"]]),
                                 issues_keys))

    # Split the issues
    n_issues = len(issues_keys)
    training_issues = set(random.sample(list(issues_keys), k=round(n_issues * split)))

    data_file = Path(data_file)
    with data_file.open("r") as f:
        entries = json.load(f)

    # Place the entries in their respective set, according to the split
    training_set = []
    eval_set = []
    for entry in entries:
        if entry["issue"] in training_issues:
            training_set.append(entry)
        elif entry["issue"] in issues_keys:
            eval_set.append(entry)

    output_path = Path(output_dir)
    with output_path.joinpath("train.json").open("w") as f:
        f.write(json.dumps(training_set))

    with output_path.joinpath("eval.json").open("w") as f:
        f.write(json.dumps(eval_set))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--project_file", default=None, type=str,
                        help="The input training data file (a json file).")
    parser.add_argument("--data_file", default=None, type=str,
                        help="The input training data file (a json file).")
    parser.add_argument("--duplicate_file", default=None, type=str,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The input training data file (a json file).")
    parser.add_argument("--seed", default=None, type=int,
                        help="The input training data file (a json file).")
    parser.add_argument("--split", default=None, type=float,
                        help="The input training data file (a json file).")

    args = parser.parse_args()

    split(args.project_file, args.duplicate_file, args.data_file, args.output_dir)


if __name__ == "__main__":
    main()
