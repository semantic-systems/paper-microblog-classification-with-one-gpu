import argparse


def parse():
    parser = argparse.ArgumentParser(description=' and the job name.')
    parser.add_argument('config', metavar='C', type=str,
                        help='Path to a configuration file (.yaml)')
    parser.parse_args()
    return parser.parse_args()