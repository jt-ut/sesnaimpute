"""One command line per {region, class} fit job, for a cluster
(IMPLEMENTATION_BMSTP_DRAFT.md section 4 row 2.7): each line runs one
region's one class's sweep (`fittp.sweep.build`) in its own capped
process, the pattern of `RUNBOOK.sh`'s own `--- fit ---` block (`fit.run
--jobs`'s `bms/fit/jobs.sh` line).
"""

import sys

from sesnaimpute import config as config_module
from sesnaimpute import regions as regions_module
from sesnaimpute.fittp.sweep import CLASSES


def jobs(config_path):
    """Yields one `./capped.sh ... sesnaimpute.fittp.sweep <config> --regions
    "<region>" --classes <cls>` line per {region, class} pair, all thirty
    regions, all six classes."""
    for region in [r.name for r in regions_module.REGIONS]:
        for cls in CLASSES:
            yield ('./capped.sh /usr/local/bin/python3.9 -m sesnaimpute.fittp.sweep %s '
                   '--regions "%s" --classes %s' % (config_path, region, cls))


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_module.load(config_path)  # fail loudly if the config itself is bad
    for line in jobs(config_path):
        print(line)
