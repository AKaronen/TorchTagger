from __future__ import annotations

from jsonargparse import ArgumentParser
import os
from typing import Any, Dict, Tuple

import yaml
import re


def parse_cli() -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Parse command line args, load YAML config, and return merged config.

    Uses `jsonargparse.ArgumentParser` to provide nicer help and subcommand
    handling while still loading the YAML manually so arbitrary keys are
    preserved.
    """
    parser = ArgumentParser(prog="tagger")

    subcommands = parser.add_subcommands(dest="mode", required=True)

    # train parser
    train_p = ArgumentParser()
    train_p.add_argument("-b", "--batch-size", type=int)
    train_p.add_argument("-e", "--epochs", type=int)
    train_p.add_argument("--lr", type=float)
    train_p.add_argument("-p", "--percentage", type=float)
    train_p.add_argument("--validation-split", type=float)
    train_p.add_argument("--data", type=str, help="Path to data folder")
    train_p.add_argument(
        "-n", "--np-data", action="store_true", help="Use numpy data loader"
    )
    train_p.add_argument("--resume-training", action="store_true")
    train_p.add_argument("--ckpt-path", type=str)
    train_p.add_argument("-o", "--output", type=str)
    train_p.add_argument("--model", type=str)
    train_p.add_argument("-c", "--config", required=True, help="Path to YAML config")
    train_p.add_argument(
        "--logger", action="store_true", help="Enable logger (TensorBoard)"
    )
    subcommands.add_subcommand("train", parser=train_p)

    # test parser
    test_p = ArgumentParser()
    test_p.add_argument("-b", "--batch-size", type=int)
    test_p.add_argument("--data", type=str)
    test_p.add_argument("--ckpt-path", type=str, required=True)
    test_p.add_argument("-o", "--output", type=str)
    test_p.add_argument("-c", "--config", required=True, help="Path to YAML config")
    subcommands.add_subcommand("test", parser=test_p)

    parsed = parser.parse_args()
    mode = parsed.mode
    parsed = getattr(parsed, mode)
    # load YAML config manually so arbitrary keys are available
    cfg_path = os.path.abspath(parsed.config)
    if not os.path.exists(cfg_path):
        parser.error(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        cfg: Dict[str, Any] = yaml.load(f, Loader=loader) or {}

    extras: Dict[str, Any] = {}

    # apply overrides similar to before
    if mode == "train":
        training_cfg = cfg.setdefault("training_config", {})
        data_cfg = cfg.setdefault("data_config", {})

        if parsed.batch_size is not None:
            data_cfg["batch_size"] = int(parsed.batch_size)
        if parsed.epochs is not None:
            training_cfg["epochs"] = int(parsed.epochs)
        if parsed.lr is not None:
            if "optimizer_params" in training_cfg and isinstance(
                training_cfg["optimizer_params"], dict
            ):
                training_cfg.setdefault("optimizer_params", {})["lr"] = float(parsed.lr)
            else:
                training_cfg["lr"] = float(parsed.lr)
        if parsed.percentage is not None:
            data_cfg["percentage"] = float(parsed.percentage)
        if parsed.validation_split is not None:
            data_cfg["validation_split"] = float(parsed.validation_split)
        if parsed.data is not None:
            data_cfg["data_path"] = parsed.data
            cfg.setdefault("hparams", {})["data"] = parsed.data
        if parsed.np_data:
            data_cfg["np_data"] = True
            cfg.setdefault("hparams", {})["np_data"] = True
        if parsed.output is not None:
            cfg["output"] = parsed.output
        if parsed.model is not None:
            cfg["model"] = parsed.model

        extras["resume_training"] = bool(parsed.resume_training)
        extras["ckpt_path"] = parsed.ckpt_path

    elif mode == "test":
        data_cfg = cfg.setdefault("data_config", {})
        if getattr(parsed, "batch_size", None) is not None:
            cfg.setdefault("data_config", {})["batch_size"] = int(parsed.batch_size)
        if getattr(parsed, "data", None) is not None:
            data_cfg["data_path"] = parsed.data
        if getattr(parsed, "output", None) is not None:
            cfg["output"] = parsed.output
        extras["ckpt_path"] = getattr(parsed, "ckpt_path", None)

    return mode, cfg, extras
