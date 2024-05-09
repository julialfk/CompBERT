# import sys
import argparse

def collect_reqs(args) -> dict[str, str]:
    # TODO

def add_reqs():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--methods_path", default=None, type=str,
                        help="The path to the json containing the project's methods.")
    parser.add_argument("--requirements_path", default=None, type=str,
                        help="The path to the directory containing the project's requirements.")
    parser.add_argument("--traces_path", default=None, type=str,
                        help="The path to the directory containing the project's trace link xml files.")
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="The output directory where the preprocessed data will be written.")

    #print arguments
    args = parser.parse_args()

    reqs_dict = collect_reqs(args)

if __name__ == "__main__":
    add_reqs()