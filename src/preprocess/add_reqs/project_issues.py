import argparse
import json
from pathlib import Path


def format_query(args):
    query_path = Path(args.query_path)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with query_path.open("r", encoding="utf-8") as f:
        query = json.load(f)
        
    commits = query[0]["rows"]

    commits_json = dict()
    for commit in commits:
        issue_id = commit[0]
        summary = commit[1]
        description = commit[2]
        resolved_date = commit[3]
        commit_id = commit[4]

        if issue_id not in commits_json:
            commits_json[issue_id] = dict()
            commits_json[issue_id]["commits"] = []

        commits_json[issue_id]["summary"] = summary
        commits_json[issue_id]["description"] = description
        commits_json[issue_id]["resolved_date"] = resolved_date
        commits_json[issue_id]["commits"].append(commit_id)

    with output.open("w", encoding="utf-8") as o:
        o.write(json.dumps(commits_json))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--query_path",
        default=None,
        type=str,
        required=False,
        help="The path to the json containing the project's data.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=False,
        help="The path to where the preprocessed data will be written.",
    )

    # print arguments
    args = parser.parse_args()

    format_query(args)


if __name__ == "__main__":
    main()
