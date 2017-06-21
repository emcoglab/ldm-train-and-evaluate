# coding=utf-8
"""
Common code.
"""

import sys
from datetime import datetime


def prints(*args, sep=' ', end='\n', file=None, flush=False):
    """
    prints(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)

    Prints the values to a stream, or to sys.stdout by default.

    Attaches a local timestamp to the start of the output.

    :param args:
    Optional keyword arguments:
    :param file:  a file-like object (stream); defaults to the current sys.stdout.
    :param sep:   string inserted between values, default a space.
    :param end:   string appended after the last value, default a newline.
    :param flush: whether to forcibly flush the stream.
    """
    timestamp = "[{0}]".format(datetime.now())
    print(timestamp, *args, sep=sep, end=end, file=file, flush=flush)


def parse_args(args):
    """
    Parses command line arguments into switches, parameters and commands.
    Switches look like "-switch"
    Parameters look like "param=value"
    Commands look like "command" (no initial "-")

    :param args: List of strings straight from the console.
    :return switches: list of strings which are switches, leading "-" trimmed
    :return parameters: dictionary of parameters
    :return commands: list of strings which are commands
    """

    # Switches look like "-switch"
    switches = [
        arg
        for arg in args
        if arg[0] == "-"
    ]

    # Parameters look like "parameter=value"
    parameters = dict([
        (
            arg.split("=")[0],
            arg.split("=")[1]
        )
        for arg in args
        if arg[0] != "-" and "=" in arg
    ])

    # commands look like "command"
    commands = [
        arg
        for arg in args
        if arg[0] != "-" and "=" not in arg
    ]

    return switches, parameters, commands


def irange(mi, ma):
    """
    inclusive range, returns range(mi, ma+1)
    :param mi:
    :param ma:
    :return:
    """
    return range(mi, ma+1)


def Nones(l):
    """
    Produces a list of None of the specified length l
    :param l:
    :return:
    """
    return repvect(l, None)


def zeros(l):
    """
    Produces a list of 0 of the specified length l
    :param l:
    :return:
    """
    return repvect(l, 0)


def repvect(l, v):
    """
    Produces a list of v of the specified length l
    :param v:
    :param l:
    :return:
    """
    return l * [v]


def get_parameter(parameters, param_name, required=False, usage_text=None):
    """
    Gets parameters from a parameter list
    :param parameters:
    :param param_name:
    :param required:
    :param usage_text:
    :return: :raise ValueError:
    """
    if param_name in parameters:
        param = parameters[param_name]
        # Want to remove trailing spaces (shouldn't be necessary but what the
        # hell) and quotes which may exist around paths.
        param = param.strip(" ")
        param = param.strip('"')
    elif required:
        if usage_text is not None:
            print(usage_text)
        raise ValueError("Require {0} parameter.".format(param_name))
    else:
        return ""
    return param


class ApplicationError(Exception):
    """
    Represents an error in which the executing code is in a logically invalid
    state. This means that a programmer error has occurred.
    :param value:
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class InvalidOperationError(Exception):
    """
    Represents an error in which the operation is invalid given the state of things.
    :param value:
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class RedirectStdoutTo:

    """
    For redirecting stdout to a file.
    Copied from Dive Into Python.

    Use like:

    with open('log.txt', mode="w", encoding="utf-8") as log_file, RedirectStdoutTo(log_file):
        # do stuff here

    :param out_new:
    """

    def __init__(self, out_new):
        self.out_new = out_new

    def __enter__(self):
        self.out_old = sys.stdout
        sys.stdout = self.out_new

    def __exit__(self, *args):
        sys.stdout = self.out_old


if __name__ == "__main__":
    raise InvalidOperationError("Library code shouldn't be run directly.")
