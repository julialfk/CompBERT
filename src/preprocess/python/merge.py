import argparse
import json
import csv
import re
from pathlib import Path


def get_project(root_dir, project_name, excludes):
    """
    Retrieve and filter project data from the specified JSON file.

    Args:
        root_dir (Path): The root directory containing the project folders.
        project_name (str): The name of the project folder.
        excludes (list): A list of exclusion patterns to filter out
                         specific paths.

    Returns:
        list: A list of filtered data entries for the specified project.
    """
    input_path = root_dir.joinpath(project_name, "data.json")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    data_new = []
    for item in data:
        if isExcluded(item["path"], excludes):
            continue

        item["project"] = project_name
        data_new.append(item)

    return data_new


def isExcluded(path: str, excludes: list):
    """
    Determine if a given path matches any exclusion pattern.

    Args:
        path (str): The path to check.
        excludes (list): A list of exclusion patterns.

    Returns:
        bool: True if the path matches any exclusion pattern, otherwise False.
    """
    for ex in excludes:
        if re.search(ex, path) is not None:
            return True

    return False


def write_dataset(root_dir, format, excludes_base, projects, dataset):
    """
    Write the filtered dataset to a file in the specified format.

    Args:
        root_dir (Path): The root directory containing the project folders.
        format (str): The format for the output file ("json" or "csv").
        excludes_base (list): A list of base exclusion patterns.
        projects (list): A list of tuples where each tuple contains
                         a project name and exclusion patterns.
        dataset (str): The name of the dataset (e.g., "train", "dev", "eval").

    Returns:
        list: A list of all filtered data entries across all projects.
    """
    data_all = []

    for project, excludes in projects:
        excludes += excludes_base
        review_issues = get_review_issues(root_dir, project)
        data = get_project(root_dir, project, excludes)

        data_all += [entry for entry in data
                     if entry["issue"] not in review_issues]
        print(f"Total: {len(data_all)} items")

    output_path = root_dir.parent.joinpath("all",
                                           f"data_{dataset}_test.{format}")

    with output_path.open("w", encoding="utf-8") as f:
        if format == "json":
            f.write(json.dumps(list(data_all)))
        elif format == "csv":
            f_csv = csv.writer(f)
            f_csv.writerow(
                [
                    "project",
                    "issue",
                    "summary",
                    "description",
                    "commit",
                    "path",
                    "method_name",
                    "start_line",
                    "end_line",
                    "parent",
                    "nl_input",
                    "code",
                    "changed",
                ]
            )
            for item in data_all:
                f_csv.writerow(
                    [
                        item["project"],
                        item["issue"],
                        item["summary"],
                        item["description"],
                        item["commit"],
                        item["path"],
                        item["method_name"],
                        item["start_line"],
                        item["end_line"],
                        item["parent"],
                        item["nl_input"],
                        item["code"],
                        item["changed"],
                    ]
                )

    return data_all


def count(root_dir, excludes_base, projects, dataset):
    """
    Count and print statistics for the dataset.

    Args:
        root_dir (Path): The root directory containing the project folders.
        excludes_base (list): A list of base exclusion patterns.
        projects (list): A list of tuples where each tuple contains
                         a project name and exclusion patterns.
        dataset (str): The name of the dataset ("train", "dev", "eval").
    """
    for project, excludes in projects:
        excludes += excludes_base
        review_issues = get_review_issues(root_dir, project)
        data = get_project(root_dir, project, excludes)

        issues = set()
        pos = 0
        neg = 0

        data = [entry for entry in data if entry["issue"] not in review_issues]

        for entry in data:
            issues.add(entry["issue"])
            if entry["changed"] is True:
                pos += 1
            else:
                neg += 1

        print(f"dataset: {dataset}")
        print(f"{project}: {len(data)} items")
        print(f"pos: {pos}, neg: {neg}")
        print(f"n_issues: {len(issues)}")
        print()


def get_review_issues(root_dir, project):
    """
    Get the issues that are associated with invalid commits.

    Args:
        root_dir (Path): The root directory containing the project folders.
        project (str): The name of the project folder.

    Returns:
        set: A set of issues associated with invalid commits.
    """
    input_path = root_dir.joinpath(project, "review_commits.json")
    with input_path.open("r", encoding="utf-8") as f:
        review_commits = json.load(f)

    input_path = root_dir.joinpath(project, f"{project}_query.json")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    review_commits = [item[0] for item in review_commits]

    review_issues = set()
    for issue, info in data.items():
        if any(commit in review_commits for commit in info["commits"]):
            review_issues.add(issue)

    return review_issues


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="The path to the directory containing the projects.",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    excludes_base = [r"^test/", "/test/", "example/", "examples/", "mock/"]
    training_projects = [
        ("drools", ["testframework/",
                    "/TestNode.java",
                    "testing/",
                    "integrationtests/",
                    "/Tester.java",
                    "drools-example"]),
        ("groovy", ["test-resources"]),
        ("maven", ["integration-tests/",
                   "TestResourcesMojo.java",
                   "test-plugin/"]),
        ("infinispan", ["testsuite/",
                        "integrationtests/",
                        "integrationtest/"]),
        ("pig", []),
        ("seam2", ["examples-ee6/", "generated-component/"]),
    ]
    dev_projects = [
        ("keycloak", [r"^testsuite"]),
        ("cassandra", []),
        ("hornetq", [r"^tests/"]),
        ("lucene", []),
        ("hibernate", []),
        ("kafka", ["MockProducer.java"]),
        ("railo", []),
        ("jboss", ["/tests/"]),
        ("resteasy", ["TestResourceLazyValidator.java", "/tests/"]),
        ("zookeeper", ["TestClient.java"]),
    ]
    eval_projects = [
        ("errai", []),
        ("spark", []),
        ("izpack", []),
        ("hadoop", []),
        ("hbase", ["CompressionTest.java"]),
        ("teiid", []),
        ("flink", ["TestProcessingTimeService.java", "flink-gelly-examples/"]),
        ("switchyard", []),
        ("hive", []),
        ("archiva", [])
    ]

    write_dataset(root_dir, "json", excludes_base, training_projects, "train")
    count(root_dir, excludes_base, training_projects, "train")
    write_dataset(root_dir, "json", excludes_base, dev_projects, "dev")
    count(root_dir, excludes_base, dev_projects, "dev")
    write_dataset(root_dir, "json", excludes_base, eval_projects, "eval")
    count(root_dir, excludes_base, eval_projects, "eval")


if __name__ == "__main__":
    main()
