import os.path
import sys
import bz2
import gzip
import logging
import contextlib
import plac

def openz(path, mode = "rb", closing = True):
    """Open a file, transparently [de]compressing it if a known extension is present."""

    (_, extension) = os.path.splitext(path)

    if extension == ".bz2":
        file_ = bz2.BZ2File(path, mode)
    elif extension == ".gz":
        file_ = gzip.GzipFile(path, mode)
    elif extension == ".xz":
        raise NotImplementedError()
    else:
        return open(path, mode)

    if closing:
        return contextlib.closing(file_)

def static_path(relative):
    """Return the path to an associated static data file."""
    return os.path.join(os.path.dirname(__file__), "static", relative)

class TTY_Formatter(logging.Formatter):
    """A log formatter for console output."""

    _DATE_FORMAT = "%H:%M:%S"
    _TIME_COLOR  = "\x1b[34m"
    _NAME_COLOR  = "\x1b[35m"
    _LEVEL_COLOR = "\x1b[33m"
    _COLOR_END   = "\x1b[00m"

    def __init__(self, stream = None):
        """
        Construct this formatter.

        Provides colored output if the stream parameter is specified and is an acceptable TTY.
        We print hardwired escape sequences, which will probably break in some circumstances;
        for this unfortunate shortcoming, we apologize.
        """

        # select and construct format string
        import curses

        format = None

        if stream and hasattr(stream, "isatty") and stream.isatty():
            curses.setupterm()

            # FIXME do nice block formatting, increasing column sizes as necessary
            if curses.tigetnum("colors") > 2:
                format = \
                    "%s%%(asctime)s%s %%(message)s" % (
                        TTY_Formatter._NAME_COLOR,
                        TTY_Formatter._COLOR_END,
                        )

        if format is None:
            format = "%(name)s - %(levelname)s - %(message)s"

        # initialize this formatter
        logging.Formatter.__init__(self, format, TTY_Formatter._DATE_FORMAT)

def log_level_to_number(level):
    """Convert a level name to a level number, if necessary."""

    if type(level) is str:
        return logging._levelNames[level]
    else:
        return level

def get_logger(name = None, level = None, default_level = logging.INFO):
    """Get or create a logger."""

    if name is None:
        logger = logging.root
    else:
        logger = logging.getLogger(name)

    # set the default level, if the logger is new
    try:
        clean = logger.is_squeaky_clean
    except AttributeError:
        pass
    else:
        if clean and default_level is not None:
            logger.setLevel(log_level_to_number(default_level))

    # unconditionally set the logger level, if requested
    if level is not None:
        logger.setLevel(log_level_to_number(level))

        logger.is_squeaky_clean = False

    return logger

def enable_default_logging():
    """Set up logging in the typical way."""

    get_logger(level = "INFO")

    # build a handler
    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(TTY_Formatter(sys.stdout))

    # add it
    logging.root.addHandler(handler)

    return handler

def script(main):
    """Call a script main function."""

    enable_default_logging()

    plac.call(main)

annotations = plac.annotations

