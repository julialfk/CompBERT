import os
import argparse
import requests
import re
import json
from dotenv import load_dotenv
from pathlib import Path


def pull_diff(owner: str, repo: str, head: str, output_dir: str):
    output_dir = Path(__file__).parent.joinpath(output_dir, head)
    output_dir.mkdir(parents=True, exist_ok=True)

    url_head = (
        f"https://api.github.com/repos/{owner}/{repo}/commits/{head}"
    )
    payload = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.getenv("GITHUB_TOKEN")}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    r = requests.get(url_head, params=payload)
    if r.status_code != 200:
        raise Exception(f"Commit not found. Status code: {r.status_code}\n"
                         "args:\n"
                        f"\towner = {owner}\n"
                        f"\trepo  = {repo}\n"
                        f"\thead  = {head}\n")
    r_json = r.json()
    print(head)
    with (output_dir.joinpath("commit.json")).open("w", encoding="utf-8") as f:
        f.write(r.text)

    files_json = r_json["files"]

    parents = r_json["parents"]
    if len(parents) > 0:
        parent_ids = [parent["sha"] for parent in parents]

    # Check if there are modified files for which no patch message was given
    # no_patch_mods = any(file["status"] == "modified" and "patch" not in file for file in files_json)
    # if no_patch_mods:
    #     all_patches = requests.get(f"https://github.com/{owner}/{repo}/commit/{head}.patch")

    diff_lines = dict()
    for file in files_json:
        file_status = file["status"]
        file_name = file["filename"]

        # Only search modified java files
        is_java = re.search(r"\.(java|jav)", file_name)
        if ((file_status != "modified") and (file_status != "added")) or (is_java is None):
            continue

        print(file_name)

        diff_lines[file_name] = get_lines(file)
        save_file(file, output_dir)
        if file_status == "modified":
            get_parents(file, output_dir, r_json, parent_ids)

    # Save the diff lines as a json in the commit directory
    diff_path = output_dir.joinpath("diff_lines.json")
    with diff_path.open("w") as f:
        f.write(json.dumps(diff_lines))

# Save new commit file
def save_file(file, output_dir):
    new_raw = requests.get(file["raw_url"]).text
    file_path = output_dir.joinpath("new", file["filename"])
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        f.write(new_raw)

# Get files from the parents
def get_parents(file, output_dir, r_json, parent_ids):
    old_raw_urls = [file["raw_url"].replace(r_json["sha"], parent_id) for parent_id in parent_ids]

    for index, url in enumerate(old_raw_urls):
        old_r = requests.get(url)
        if old_r.status_code == 200:
            file_path = output_dir.joinpath(f"old{index}", file["filename"])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as f:
                f.write(old_r.text)

def get_lines(file):
    diff_before = []
    diff_after = []
    if "patch" in file:
        file_patch = file["patch"]

        # Find all shown lines in file patch
        diffs = re.findall(r"@@ -[0-9]+,[0-9]+ \+[0-9]+,[0-9]+ @@", file_patch)
        for diff_str in diffs:
            diff_before, diff_after = calc_lines(diff_before, diff_after, diff_str)
    elif file["status"] == "modified":
        raise Exception("file is modified, but no change lines were given.")

    # lines stay empty if file is added and number of lines is too many to show
    return {"diff_before": diff_before, "diff_after": diff_after}


def calc_lines(diff_before, diff_after, diff_str):
    diff = re.split(r" |,", diff_str[3:-3])
    diff_start_before = int(diff[0][1:])
    diff_end_before = diff_start_before + int(diff[1]) - 1
    diff_start_after = int(diff[2][1:])
    diff_end_after = diff_start_after + int(diff[3]) - 1

    diff_before.append((diff_start_before, diff_end_before))
    diff_after.append((diff_start_after, diff_end_after))

    return diff_before, diff_after


def read_commits(args):
    input_path = Path(args.input_path)
    with input_path.open("r") as f:
        issues = json.load(f)

    for issue in issues.values():

        for commit in issue["commits"]:
            # Skip commit if it has been pulled in a previous run
            if Path(__file__).parent.joinpath(args.output_dir, commit).exists():
                continue
            pull_diff(args.owner, args.repo, commit, args.output_dir)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--owner", default=None, type=str, required=True, help="The owner of the GitHub repository."
    )
    parser.add_argument(
        "--repo", default=None, type=str, required=True, help="The name of the repository."
    )
    parser.add_argument(
        "--next_issue", default="", type=str, required=False, help="The ID of the next issue to pull from GitHub."
    )
    # parser.add_argument(
    #     "--head", default=None, type=str, required=True, help="The commit tag of the child."
    # )
    # parser.add_argument(
    #     "--token", default=None, type=str, required=True, help="The Github personal access token."
    # )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the preprocessed data will be written.",
    )
    parser.add_argument(
        "--input_path",
        default=None,
        type=str,
        required=True,
        help="The path to the json containing the issues and their commits' information.",
    )

    # print arguments
    args = parser.parse_args()

    load_dotenv()
    read_commits(args)


if __name__ == "__main__":
    main()
