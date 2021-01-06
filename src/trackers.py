import ncc
import mosse
import multi_channel_mosse

def add_tracker_args(argument_parser):
    argument_parser.add_argument("--tracker", choices=("NCC", "MOSSE", "MCM"), required=True)
    ncc.add_cli_arguments(argument_parser)
    mosse.add_cli_arguments(argument_parser)
    multi_channel_mosse.add_cli_arguments(argument_parser)


def get_tracker(args):
    if args.tracker == "NCC":
        tracker = ncc.get_tracker(args)
    elif args.tracker == "MOSSE":
        tracker = mosse.get_tracker(args)
    elif args.tracker == "MCM":
        tracker = multi_channel_mosse.get_multichannel_tracker(args)
    else:
        raise RuntimeError("Invalid tracker")

    return tracker