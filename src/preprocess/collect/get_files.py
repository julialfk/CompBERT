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


def format_query(query_path: str, project_dir: str):
    """
    Processes a query JSON file containing commit information,
    formats the data, and saves the result to a new JSON file.

    Args:
        args: An argparse.Namespace object containing the following attributes:
            - query_path (str): The path to the input query JSON file.
            - project_dir (str): The directory where the output JSON file
                                 should be saved.

    Returns:
        dict: A dictionary where keys are issue IDs and values are
              dictionaries containing issue details and related commits.
    """
    query_path = Path(query_path)
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    with query_path.open("r", encoding="utf-8") as f:
        query = json.load(f)

    commits = query[0]["rows"]

    issues = dict()
    for commit in commits:
        issue_id = commit[0]
        summary = re.sub(r"^\[[A-Z]+\-[0-9]+\]\s*", "", commit[1])
        description = commit[2]
        commit_id = commit[3]

        if issue_id not in issues:
            issues[issue_id] = dict()
            issues[issue_id]["commits"] = []

        # Merge the summary and description into single nl input
        description_soup = BeautifulSoup(description, "html5lib")
        for div in description_soup.find_all("div", {"class": "code panel"}):
            div.decompose()
        nl_input = summary + "\n" + description_soup.get_text()

        # Store the processed data in the issues dictionary
        issues[issue_id]["nl_input"] = nl_input
        issues[issue_id]["summary"] = summary
        issues[issue_id]["description"] = description
        issues[issue_id]["commits"].append(commit_id)

    # Save the formatted issues dictionary to a new JSON file
    save_json(project_dir.joinpath(f"{project_dir.name}_query.json"), issues)

    return issues


def get_diff(owner: str, repo: str, head: str, project_dir: str,
             review_commits: set[tuple]):
    """
    Retrieves the diff of a specific commit from a GitHub repository and saves
    the commit information and diff lines.

    Args:
        owner (str): The owner of the GitHub repository.
        repo (str): The name of the repository.
        head (str): The commit SHA to pull the diff for.
        project_dir (str): The directory where the commit information
                           will be saved.
        review_commits (set[tuple]): A set to store commits that need review.

    Returns:
        set[tuple]: The updated set of commits that need review.
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
        save_json(commit_dir.parent.joinpath("review_commits.json"),
                  list(review_commits))

        msg = f"Commit not found. Status code: {r.status_code}"
        review_commits = mark_commit_for_review(commit_dir, review_commits,
                                                head, msg)
        if r.status_code != 422:
            raise Exception(msg)

        return review_commits

    r_commit = r.json()
    logger.info(f"commit: {head}")
    save_json(commit_dir.joinpath("commit.json"), r.text)

    files = r_commit["files"]

    parents = r_commit["parents"]
    if len(parents) == 1:
        parent_id = parents[0]["sha"]
    elif len(parents) > 1:
        return mark_commit_for_review(commit_dir, review_commits, head,
                                      "Commit has more than one parent.")
    else:
        return mark_commit_for_review(commit_dir, review_commits, head,
                                      "Commit has no parents.")

    # Request the .patch
    r_patches = requests.get(
        f"https://github.com/{owner}/{repo}/commit/{head}.patch"
    )
    if r_patches.status_code != 200:
        msg = f"Error when requesting the .patch: {r_patches.status_code}"
        return mark_commit_for_review(commit_dir, review_commits, head, msg)

    patches = split_patches(r_patches.text)

    # Extract diff lines for modified or added Java files
    diff_lines, review_commits = process_files(files, patches, owner, repo,
                                               head, commit_dir, parent_id,
                                               review_commits)

    if diff_lines is not None:
        add_diff_lines(commit_dir, diff_lines)

    return review_commits


def process_files(files_json, patches, owner, repo, head, commit_dir,
                  parent_id, review_commits):
    """
    Process the files in the commit to extract and save their diff lines.

    Args:
        files_json (list): List of files in the commit.
        patches (dict): Dictionary of patches for the files.
        owner (str): The owner of the GitHub repository.
        repo (str): The name of the repository.
        head (str): The commit SHA.
        commit_dir (Path): The directory where the commit information is saved.
        parent_id (str): The SHA of the parent commit.
        review_commits (set[tuple]): A set to store commits that need review.

    Returns:
        dict: A dictionary containing the diff lines for each file.
        set[tuple]: The updated set of commits that need review.
    """
    diff_lines = dict()
    for file in files_json:
        file_status = file["status"]
        file_name = file["filename"]

        # Only process modified or added Java files
        is_java = re.search(r"\.(java|jav)$", file_name) is not None
        if not is_java or file_status not in {"modified", "added"}:
            continue

        logger.info(file_name)

        try:
            diff_lines[file_name] = get_lines(patches[file_name])
        except KeyError:
            return (None,
                    mark_commit_for_review(commit_dir, review_commits, head,
                                           "Commit does not include a patch "
                                           "for this file, due to e.g. "
                                           "the creation of an empty file."))

        save_file(file, commit_dir)
        if file_status == "modified":
            status_code, review_commits = get_parents(
                owner, repo, head, commit_dir, parent_id,
                diff_lines[file_name]["old_file"], review_commits
            )
            if status_code != 200:
                return None, review_commits

    return diff_lines, review_commits


def add_diff_lines(commit_dir: Path, content):
    """
    Save the diff lines as a JSON file in the commit directory.

    Args:
        commit_dir (Path): The directory where the commit information is saved.
        content: The diff lines content to be saved.
    """
    diff_path = commit_dir.joinpath("diff_lines.json")
    with diff_path.open("w") as f:
        f.write(json.dumps(content))


def save_json(file_path, data):
    """
    Save data as a JSON file.

    Args:
        file_path (Path): The path to the JSON file.
        data (dict): The data to be saved as JSON.
    """
    with file_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(data))


def mark_commit_for_review(commit_dir, review_commits, head, msg):
    """
    Mark a commit for review by saving the message and updating
    the review_commits set.

    Args:
        commit_dir (Path): The directory where the commit information is saved.
        review_commits (set[tuple]): A set to store commits that need review.
        head (str): The commit SHA.
        msg (str): The message explaining why the commit needs review.
    """
    add_diff_lines(commit_dir, msg)
    review_commits.add((head, msg))

    return review_commits


def split_patches(patches: str):
    """
    Split a patch string into individual patches for each file.

    Args:
        patches (str): The patch string containing diff information.

    Returns:
        dict: A dictionary where keys are file names and values
              are patch details.
    """
    patches = re.split("diff --git", patches)

    patches_formatted = dict()
    for patch in patches[1:]:
        new_file = re.search(r"\n\+\+\+ ([^\n])+\.(java|jav)\n", patch)
        if not new_file:
            continue

        new_file = new_file.group()[7:-1]
        old_file = re.search(r"\n\-\-\- ([^\n])+", patch).group()[5:]
        if old_file != "/dev/null":
            old_file = old_file[2:]

        patch_content = re.split(
            r"@@ -[0-9]+,[0-9]+ \+[0-9]+,[0-9]+ @@[^\n]*\n", patch
        )[1:]
        patch_lines = re.findall(
            r"@@ -[0-9]+,[0-9]+ \+[0-9]+,[0-9]+ @@", patch
        )

        patches_formatted[new_file] = {"old_file": old_file,
                                       "patches": list(zip(patch_lines,
                                                           patch_content))}

    return patches_formatted


def save_file(file: dict, commit_dir: Path):
    """
    Saves the raw content of a file from a commit to the specified directory.

    Args:
        file (dict): A dictionary containing file information including
                     the raw URL.
        commit_dir (Path): The directory where the file should be saved.
    """
    new_raw = requests.get(file["raw_url"]).text
    file_path = commit_dir.joinpath("new", file["filename"])
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        f.write(new_raw)


def get_parents(owner: str, repo: str, head: str, commit_dir: Path,
                parent_id: str, old_file_name: str, review_commits: set[str]):
    """
    Retrieves and saves the parent versions of a modified file.

    Args:
        owner (str): The owner of the GitHub repository.
        repo (str): The name of the repository.
        head (str): The commit SHA.
        commit_dir (Path): The directory where the file should be saved.
        parent_id (str): The SHA of the parent commit.
        old_file_name (str): The name of the file in the parent commit.
        review_commits (set[str]): A set of commits that need review.

    Returns:
        int: The HTTP status code of the request to retrieve the parent file.
        set[str]: The updated set of commits that need review.
    """
    old_raw_url = (
        f"https://github.com/{owner}/{repo}/raw/{parent_id}/{old_file_name}"
    )

    old_r = requests.get(old_raw_url)
    if old_r.status_code == 200:
        file_path = commit_dir.joinpath(f"old", old_file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            f.write(old_r.text)
    else:
        msg = ("Unable to request parent file.\n"
               f"Status code: {old_r.status_code}\n"
               f"Requested url: {old_raw_url}\n")
        review_commits = mark_commit_for_review(commit_dir, review_commits,
                                                head, msg)

    return old_r.status_code, review_commits


def get_lines(patches: dict):
    """
    Extracts the lines of code that have been changed in a file patch.

    Args:
        patches (dict): A dictionary containing patch details for a file.

    Returns:
        dict: A dictionary with keys 'diff_before' and 'diff_after' containing
              the line numbers of each patch window before and after
              the changes, and 'old_file' containing the name of the old file.
    """
    diff_before, diff_after = [], []

    for patch_lines, patch_content in patches["patches"]:
        patch_content = re.split("\n", patch_content)
        old_idx, new_idx = calc_lines(patch_lines)
        old_idx -= 1
        new_idx -= 1

        in_change_window = False
        for line in patch_content:
            if not line:
                old_idx += 1
                new_idx += 1
                # Close the last entry with the end line
                if in_change_window:
                    in_change_window = False
                    diff_before[-1].append(old_idx)
                    diff_after[-1].append(new_idx)
            elif line[0] in "+-":
                # Add new entry containing the start line only
                if not in_change_window:
                    in_change_window = True
                    diff_before.append([old_idx])
                    diff_after.append([new_idx])
                if line[0] == "+":
                    new_idx += 1
                else:
                    old_idx += 1
            else:
                old_idx += 1
                new_idx += 1
                if in_change_window:
                    in_change_window = False
                    diff_before[-1].append(old_idx)
                    diff_after[-1].append(new_idx)

        if in_change_window:
            in_change_window = False
            diff_before[-1].append(old_idx)
            diff_after[-1].append(new_idx)

    return {"diff_before": diff_before,
            "diff_after": diff_after,
            "old_file": patches["old_file"]}


def calc_lines(patch_lines: str):
    """
    Extracts the line numbers before and after a change from a diff string.

    Args:
        patch_lines (str): The diff string containing
                           the line number information.

    Returns:
        tuple: The starting line numbers before and after the change.
    """
    diff = re.split(r" |,", patch_lines[3:-3])
    diff_start_before = int(diff[0][1:])
    diff_start_after = int(diff[2][1:])

    return diff_start_before, diff_start_after


def read_commits(args, issues: dict=None):
    """
    Reads commit information for issues from a JSON file and gets the diffs
    for each commit.

    Args:
        args (argparse.Namespace): An argparse.Namespace object containing
                                   the following attributes:
            - owner (str): The owner of the GitHub repository.
            - repo (str): The name of the repository.
            - project_dir (str): The directory where the commit information
                                 will be saved.
            - issues_info_path (str): The path to the JSON file containing
                                      issue and commit information.
        issues (dict, optional): A dictionary of issues and their associated
                                 commits. If not provided, it will be read
                                 from the issues_info_path.
    """
    if issues is None:
        input_path = Path(args.issues_info_path)
        with input_path.open("r", encoding="utf-8") as f:
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

                if isinstance(files, str):
                    review_commits.add((commit, files))

                continue

            review_commits = get_diff(args.owner, args.repo, commit,
                                      args.project_dir, review_commits)

    save_json(project_dir.joinpath("review_commits.json"),
              list(review_commits))


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
        help=("The output directory where all commit and "
              "dataset information will be written."),
    )
    parser.add_argument(
        "--issues_info_path",
        default=None,
        type=str,
        required=False,
        help=("The path to the reformatted query json containing the issues "
              "and their commits in hierarchical structure."),
    )

    args = parser.parse_args()
    logging.basicConfig(format=("%(asctime)s - %(levelname)s - %(name)s -   "
                                "%(message)s"),
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

    load_dotenv()
    if args.query_path:
        issues_info = format_query(args.query_path, args.project_dir)
        read_commits(args, issues_info)
    else:
        read_commits(args)


if __name__ == "__main__":
    main()
