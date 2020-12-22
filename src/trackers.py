import ncc
import mosse


def add_tracker_args(argument_parser):
    argument_parser.add_argument("--tracker", choices=("NCC", "MOSSE"), required=True)
    ncc.add_cli_arguments(argument_parser)
    mosse.add_cli_arguments(argument_parser)


def get_tracker(args):
    if args.tracker == "NCC":
        tracker = ncc.get_tracker(args)
    elif args.tracker == "MOSSE":
        tracker = mosse.get_tracker(args)
    else:
        raise RuntimeError("Invalid tracker")

    return tracker