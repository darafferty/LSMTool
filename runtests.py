#! /usr/bin/env python

import sys

if __name__ == "__main__":
    if sys.version_info >= (3, 0):
        exec("def do_exec(co, loc): exec(co, loc)\n")
    else:
        import cPickle as pickle
        exec("def do_exec(co, loc): exec co in loc\n")

    entry = "import pytest; raise SystemExit(pytest.cmdline.main())"
    do_exec(entry, locals()) # noqa
