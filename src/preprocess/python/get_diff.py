import os
import argparse
import logging
import requests
import re
import json
from dotenv import load_dotenv
from pathlib import Path
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def format_query(args):
    """
    Processes a query JSON file containing commit information, formats the data, and saves the result to a new JSON file.

    Args:
        args: An argparse.Namespace object containing the following attributes:
            - query_path (str): The path to the input query JSON file.
            - project_dir (str): The directory where the output JSON file should be saved.
            - repo (str): The name of the repository (used to name the output file).

    Returns:
        dict: A dictionary where keys are issue IDs and values are dictionaries containing issue details and related commits.
    """
    query_path = Path(args.query_path)
    project_dir = Path(args.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    with query_path.open("r", encoding="utf-8") as f:
        query = json.load(f)

    commits = query[0]["rows"]

    issues = dict()
    for commit in commits:
        issue_id = commit[0]
        summary = re.sub(r"^\[[A-Z]+\-[0-9]+\]", "", commit[1])
        description = commit[2]
        resolved_date = commit[3]
        commit_id = commit[4]

        if issue_id not in issues:
            issues[issue_id] = dict()
            issues[issue_id]["commits"] = []

        # Merge the summary and description into single nl input
        description_soup = BeautifulSoup(description, "html.parser")
        for div in description_soup.find_all("div", {"class": "code panel"}):
            div.decompose()
        nl_input = summary + "\n" + description_soup.get_text()

        # Store the processed data in the issues dictionary
        issues[issue_id]["nl_input"] = nl_input
        issues[issue_id]["summary"] = summary
        issues[issue_id]["description"] = description
        issues[issue_id]["resolved_date"] = resolved_date
        issues[issue_id]["commits"].append(commit_id)

    # Save the formatted issues dictionary to a new JSON file
    output_path = project_dir.joinpath(f"{args.repo}_query.json")
    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(issues))

    return issues


def get_diff(owner: str, repo: str, head: str, project_dir: str, review_commits: set[str]):
    """
    Retrieves the diff of a specific commit from a GitHub repository and saves the commit information and diff lines.

    Args:
        owner: The owner of the GitHub repository.
        repo: The name of the repository.
        head: The commit SHA to pull the diff for.
        project_dir: The directory where the commit information will be saved.
        review_commits: A set to store commits that need review.

    Returns:
        set[str]: The updated set of commits that need review.
    """
    commit_dir = Path(project_dir).joinpath(head)
    commit_dir.mkdir(parents=True, exist_ok=True)

    url_head = (
        f"https://api.github.com/repos/{owner}/{repo}/commits/{head}"
    )
    payload = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.getenv("GITHUB_TOKEN2")}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    r = requests.get(url_head, params=payload)
    if r.status_code != 200:
        review_path = commit_dir.parent.joinpath("review_commits.json")
        with review_path.open("w") as f:
            f.write(json.dumps(list(review_commits)))

        raise Exception(f"Commit not found. Status code: {r.status_code}\n"
                         "args:\n"
                        f"\towner = {owner}\n"
                        f"\trepo  = {repo}\n"
                        f"\thead  = {head}\n")
    r_json = r.json()
    logger.info(f"commit: {head}")
    with (commit_dir.joinpath("commit.json")).open("w", encoding="utf-8") as f:
        f.write(r.text)

    files_json = r_json["files"]

    parents = r_json["parents"]
    if len(parents) > 0:
        parent_ids = [parent["sha"] for parent in parents]

    # Check if there are modified files for which no patch message was given
    # no_patch_mods = any(file["status"] == "modified" and "patch" not in file for file in files_json)
    # if no_patch_mods:
    #     all_patches = requests.get(f"https://github.com/{owner}/{repo}/commit/{head}.patch")

    # Extract diff lines for modified or added Java files
    diff_lines = dict()
    for file in files_json:
        file_status = file["status"]
        file_name = file["filename"]

        # Only search modified or added Java files
        is_java = re.search(r"\.(java|jav)", file_name)
        if ((file_status != "modified") and (file_status != "added")) or (is_java is None):
            continue

        logger.info(file_name)

        diff_lines[file_name] = get_lines(file)
        if diff_lines[file_name]["diff_before"] == "review":
            review_commits.add(head)

        save_file(file, commit_dir)
        if file_status == "modified":
            get_parents(file, commit_dir, r_json, parent_ids)

    # Save the diff lines as a JSON in the commit directory
    diff_path = commit_dir.joinpath("diff_lines.json")
    with diff_path.open("w") as f:
        f.write(json.dumps(diff_lines))

    return review_commits

def save_file(file: dict, commit_dir: Path):
    """
    Saves the raw content of a file from a commit to the specified directory.

    Args:
        file: A dictionary containing file information including the raw URL.
        commit_dir: The directory where the file should be saved.
    """
    new_raw = requests.get(file["raw_url"]).text
    file_path = commit_dir.joinpath("new", file["filename"])
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        f.write(new_raw)

def get_parents(file: dict, commit_dir: Path, r_json: dict, parent_ids: list):
    """
    Retrieves and saves the parent versions of a modified file.

    Args:
        file: A dictionary containing file information.
        commit_dir: The directory where the file should be saved.
        r_json: The JSON response from the GitHub API containing commit information.
        parent_ids: A list of parent commit SHAs.
    """
    old_raw_urls = [file["raw_url"].replace(r_json["sha"], parent_id) for parent_id in parent_ids]

    for index, url in enumerate(old_raw_urls):
        old_r = requests.get(url)
        if old_r.status_code == 200:
            file_path = commit_dir.joinpath(f"old{index}", file["filename"])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as f:
                f.write(old_r.text)

def get_lines(file: dict):
    """
    Extracts the lines of code that have been changed in a file patch.

    Args:
        file: A dictionary containing file information including the patch.

    Returns:
        dict: A dictionary with keys 'diff_before' and 'diff_after' containing
              the line numbers of each patch window before and after the changes.
    """
    diff_before = []
    diff_after = []
    if "patch" in file:
        file_patch = file["patch"]

        # Find all shown lines in file patch
        diffs = re.findall(r"@@ -[0-9]+,[0-9]+ \+[0-9]+,[0-9]+ @@", file_patch)
        for diff_str in diffs:
            diff_before, diff_after = calc_lines(diff_before, diff_after, diff_str)
    # File is modified, but no patch is given (usually due to large changes)
    elif file["status"] == "modified":
        return {"diff_before": "review", "diff_after": "review"}

        # doubles_path = output_dir.joinpath("double_commits.json")
        # with doubles_path.open("w") as f:
        #     f.write(json.dumps(list(double_commits)))

        # raise Exception("file is modified, but no change lines were given.")

    # lines stay empty if file is added and number of lines is too many to show
    return {"diff_before": diff_before, "diff_after": diff_after}


def calc_lines(diff_before: list, diff_after: list, diff_str: str):
    """
    Extracts the line numbers before and after a change from a diff string.

    Args:
        diff_before: A list to store line number ranges before the change.
        diff_after: A list to store line number ranges after the change.
        diff_str: The diff string containing the line number information.

    Returns:
        tuple: Updated lists of line number ranges before and after the change.
    """
    diff = re.split(r" |,", diff_str[3:-3])
    diff_start_before = int(diff[0][1:])
    diff_end_before = diff_start_before + int(diff[1]) - 1
    diff_start_after = int(diff[2][1:])
    diff_end_after = diff_start_after + int(diff[3]) - 1

    diff_before.append((diff_start_before, diff_end_before))
    diff_after.append((diff_start_after, diff_end_after))

    return diff_before, diff_after


def read_commits(args, issues: dict=None):
    """
    Reads commit information for issues from a JSON file and gets the diffs for each commit.

    Args:
        args: An argparse.Namespace object containing the following attributes:
            - owner (str): The owner of the GitHub repository.
            - repo (str): The name of the repository.
            - project_dir (str): The directory where the commit information will be saved.
            - issues_info_path (str): The path to the JSON file containing issue and commit information.
        issues (optional): A dictionary of issues and their associated commits. If not provided, it will be read from the issues_info_path.
    """
    if not issues:
        input_path = Path(args.issues_info_path)
        with input_path.open("r") as f:
            issues = json.load(f)

    project_dir = Path(args.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    review_commits = set()
    for issue in issues.values():
        for commit in issue["commits"]:
            commit_dir = project_dir.joinpath(commit)
            # Skip commit if it has been pulled in a previous run
            if commit_dir.exists():
                # Check if commit has been marked for review
                with commit_dir.joinpath("diff_lines.json").open("r") as f:
                    files = json.load(f)

                for diffs in files.values():
                    if diffs["diff_before"] == "review":
                        review_commits.add(commit)
                        break

                continue

            review_commits = get_diff(args.owner, args.repo, commit, args.project_dir, review_commits)
            # review_commits.union(review_commits_update)

    review_path = project_dir.joinpath("review_commits.json")
    with review_path.open("w") as f:
        f.write(json.dumps(list(review_commits)))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--owner",
        default=None,
        type=str,
        required=True,
        help="The owner of the GitHub repository."
    )
    parser.add_argument(
        "--repo",
        default=None,
        type=str,
        required=True,
        help="The name of the repository."
    )

    parser.add_argument(
        "--query_path",
        default=None,
        type=str,
        required=False,
        help="The path to the json containing the base sql query output.",
    )

    parser.add_argument(
        "--project_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where all commit and dataset information will be written.",
    )
    parser.add_argument(
        "--issues_info_path",
        default=None,
        type=str,
        required=False,
        help="The path to the reformatted query json containing the issues and their commits in hierarchical structure.",
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)

    load_dotenv()
    if args.query_path:
        issues_info = format_query(args)
        read_commits(args, issues_info)
    else:
        read_commits(args)


if __name__ == "__main__":
    main()
