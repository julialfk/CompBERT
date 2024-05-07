import os
import argparse
import requests
import re
import json
from dotenv import load_dotenv
from pathlib import Path


def pull_diff(args):
    output_dir = Path(__file__).parent.joinpath(args.output_dir, args.head)

    url_head = (
        f"https://api.github.com/repos/{args.owner}/{args.repo}/commits/{args.head}"
    )
    payload = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.getenv("GITHUB_TOKEN")}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    r = requests.get(url_head, params=payload)
    if r.status_code != 200:
        raise Exception("Commit not found. Are the arguments correct?")
    r_json = r.json()

    files_json = r_json["files"]

    parents = r_json["parents"]
    if len(parents) > 0:
        parent_ids = [parent["sha"] for parent in parents]

    files_data = {}
    for file in files_json:
        file_status = file["status"]
        file_name = file["filename"]

        # Only search modified java files
        is_java = re.search(r"\.(java|jav)", file_name)
        if ((file_status != "modified") and (file_status != "added")) or (is_java == None):
            continue

        # Save new commit file
        raw_url = file["raw_url"]
        new_raw = requests.get(raw_url).text
        file_path = output_dir.joinpath("new", file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        files_data[file_name] = {"new_raw": str(file_path)}
        with file_path.open("w") as f:
            f.write(new_raw)

        # Get files from the parents
        if file_status == "modified":
            old_raw_urls = [re.sub(file["sha"], parent_id, raw_url) for parent_id in parent_ids]
            old_raws = []

            for index, url in enumerate(old_raw_urls):
                old_r = requests.get(url)
                if old_r.status_code == 200:
                    file_path = output_dir.joinpath(f"old{index}", file_name)
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    old_raws.append(str(file_path))
                    with file_path.open("w") as f:
                        f.write(old_r.text)

            files_data[file_name]["old_raws"] = old_raws

    # Save file paths
    json_dir = output_dir.joinpath("file_data.json")
    with json_dir.open("w") as f:
        f.write(json.dumps(files_data))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--owner", default=None, type=str, required=True, help="The owner of the GitHub repository."
    )
    parser.add_argument(
        "--repo", default=None, type=str, required=True, help="The name of the repository."
    )
    # parser.add_argument(
    #     "--base", default=None, type=str, required=True, help="The commit tag of the parent."
    # )
    parser.add_argument(
        "--head", default=None, type=str, required=True, help="The commit tag of the child."
    )
    # parser.add_argument(
    #     "--token", default=None, type=str, required=True, help="The Github personal access token."
    # )
    parser.add_argument(
        "--output_dir",
        default="tmp",
        type=str,
        required=False,
        help="The output directory where the preprocessed data will be written.",
    )

    # print arguments
    args = parser.parse_args()

    load_dotenv()
    pull_diff(args)


if __name__ == "__main__":
    main()
