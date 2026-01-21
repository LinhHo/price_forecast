import argparse


def get_runtime_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Size of each batch",
    )

    args, _ = parser.parse_known_args()
    return args
