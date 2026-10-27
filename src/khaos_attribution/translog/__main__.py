"""python -m khaos_attribution.translog verify <target> [--mirror ...] [--key ...]: the same as khaos-log-verify."""

import sys

from khaos_attribution.translog.verify import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] != "verify":
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(args[1:]))
