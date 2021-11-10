import logging
import sys

logger = logging.getLogger("covasim_aus_schools")
debug_handler = logging.StreamHandler(sys.stdout)  # info_handler will handle all messages below WARNING sending them to STDOUT
info_handler = logging.StreamHandler(sys.stdout)  # info_handler will handle all messages below WARNING sending them to STDOUT
warning_handler = logging.StreamHandler(sys.stderr)  # warning_handler will send all messages at or above WARNING to STDERR

debug_handler.setLevel(0)  # Handle all lower levels - the output should be filtered further by setting the logger level, not the handler level
info_handler.setLevel(logging.INFO)  # Handle all lower levels - the output should be filtered further by setting the logger level, not the handler level
warning_handler.setLevel(logging.WARNING)

debug_handler.addFilter(type("ThresholdFilter", (object,), {"filter": lambda x, logRecord: logRecord.levelno < logging.INFO})())  # Display anything INFO or higher
info_handler.addFilter(type("ThresholdFilter", (object,), {"filter": lambda x, logRecord: logRecord.levelno < logging.WARNING})())  # Don't display WARNING or higher

debug_formatter = logging.Formatter("%(levelname)s {%(filename)s:%(lineno)d} - %(message)s")
debug_handler.setFormatter(debug_formatter)

logger.addHandler(debug_handler)
logger.addHandler(info_handler)
logger.addHandler(warning_handler)
logger.setLevel("INFO")  # Set the overall log level

# from .analyze_clusters import *
from .interventions import *
from .utils import *
from .population import *
from .samples import *
from .analyzers import *
from .vaccination import *
from .restrictions import *

import pathlib

datadir = pathlib.Path(__file__).parent.parent / "data"
