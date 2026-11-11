"""Run the same workshop operations from notebooks or a developer terminal."""

import argparse

from vla_workshop.course.common import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=["evaluate", "inspect", "train", "collect", "export", "compare"]
    )
    parser.add_argument("--config")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--adapter")
    parser.add_argument("--collection")
    parser.add_argument("--accepted", nargs="+", type=int)
    parser.add_argument("--destination")
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    args = parser.parse_args()
    if args.operation == "compare":
        from vla_workshop.course.compare import compare

        if not args.baseline or not args.candidate:
            parser.error("compare needs --baseline and --candidate")
        compare(args.baseline, args.candidate)
    elif args.operation == "export":
        from vla_workshop.course.collect import export_accepted

        if not args.collection or not args.accepted or not args.destination:
            parser.error("export needs --collection, --accepted, --destination")
        export_accepted(args.collection, args.accepted, args.destination)
    else:
        if not args.config:
            parser.error("This operation needs --config")
        config = read_json(args.config)
        if args.operation == "evaluate":
            from vla_workshop.course.libero import evaluate

            evaluate(config, args.split)
        elif args.operation == "inspect":
            from vla_workshop.course.data import inspect_dataset

            inspect_dataset(config)
        elif args.operation == "train":
            from vla_workshop.course.train import train

            train(config)
        elif args.operation == "collect":
            from vla_workshop.course.collect import collect

            collect(config, args.adapter)


if __name__ == "__main__":
    main()
