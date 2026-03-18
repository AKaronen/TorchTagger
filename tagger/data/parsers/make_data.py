import os


from data.parsers.h5_parser import H5DataParser
from data.parsers.huggingface_parser import HuggingFaceDataParser
from data.parsers.root_parser import RootDataParser


PARSER_REGISTRY = {
    "root": RootDataParser,
    "h5": H5DataParser,
    "hdf5": H5DataParser,
    "huggingface": HuggingFaceDataParser,
    "hf": HuggingFaceDataParser,
}


def infer_parser_name(data_config):
    explicit = (
        data_config.get("parser")
        or data_config.get("data_parser")
        or data_config.get("data_format")
        or data_config.get("dataset_type")
    )
    if explicit:
        return str(explicit).lower()

    if data_config.get("hf_dataset") or data_config.get("hf_signal"):
        return "huggingface"

    data_path = data_config.get("data_path", None)
    if data_path is None:
        return "root"

    if os.path.isfile(data_path):
        ext = os.path.splitext(data_path)[1].lower()
        if ext in (".h5", ".hdf5"):
            return "h5"
        return "root"

    if os.path.isdir(data_path):
        entries = [name.lower() for name in os.listdir(data_path)]
        if any(name.endswith((".h5", ".hdf5")) for name in entries):
            return "h5"
        if any(name.endswith(".root") for name in entries):
            return "root"

    return "root"


def get_parser(data_config):
    parser_name = infer_parser_name(data_config)
    parser_cls = PARSER_REGISTRY.get(parser_name, None)
    if parser_cls is None:
        supported = ", ".join(sorted(PARSER_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported parser '{parser_name}'. Supported parser values: {supported}"
        )
    return parser_cls(data_config)


def main(data_config):
    """Pick parser from data_config and call its make_data() implementation."""
    if not isinstance(data_config, dict):
        raise TypeError("main(data_config) expects a dict-like data_config.")

    parser = get_parser(data_config)
    print(f"Using parser: {parser.__class__.__name__}")
    return parser.make_data()


if __name__ == "__main__":
    import yaml
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Make training data from input files.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to YAML configuration file for data parsing.",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--percentage",
        type=float,
        default=None,
        help="Percentage of data to be parsed. Value between 0 and 1.",
    )
    parser.add_argument(
        "-s",
        "--test-split",
        type=float,
        default=None,
        help="Fraction of data to reserve for testing (between 0 and 1).",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/parsed/",
        help="Directory to save parsed training data.",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.percentage is not None and args.percentage < 1.0:
        config["ratio"] = args.percentage
    if args.test_split is not None and 0 < args.test_split < 1:
        config["test_split"] = args.test_split
    if args.output is not None:
        config["outdir"] = args.output

    main(config)
