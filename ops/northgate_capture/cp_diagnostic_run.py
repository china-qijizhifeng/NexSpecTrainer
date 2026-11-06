"""One instrumented real request per layout; not a production collector."""
import os

from ops.northgate_capture.retest import main

assert os.environ['NORTHGATE_CP_DIAGNOSTIC_RUN'] == '1'
main(lengths=(4096,), repeats=1)
