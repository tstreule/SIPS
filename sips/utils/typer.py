"""
Simulates tiangolo's typer module.

Required since ETH Euler does not allow to install custom modules.

"""
import argparse
import inspect

from typing_extensions import Annotated

parser = argparse.ArgumentParser()


class Typer:
    """
    Very limited pseudo-copy of tiangolo's typer module.

    """

    def __init__(self):
        self.registered_command = None
        self.kwargs_renaming = {}

    def command(self):
        # Credits: https://www.travisluong.com/advanced-python-using-decorators-argparse-and-inspect-to-build-fastarg-a-command-line-argument-parser-library/
        def wrapper(func):
            sig = inspect.signature(func)
            for name, param in sig.parameters.items():
                annotation = param.annotation
                # Ensure it's an Annotated object
                if not isinstance(annotation, Annotated):  # type: ignore[arg-type]
                    annotation = Annotated[annotation, ...]
                # Get action
                if annotation.__origin__ is bool:
                    action = argparse.BooleanOptionalAction
                else:
                    action = None
                # Get type
                if annotation.__origin__._name == "Optional":
                    type_ = annotation.__origin__.__args__[0]
                else:
                    type_ = annotation.__origin__
                # Set argument name
                if param.default is inspect._empty:
                    arg_name = name
                else:
                    arg_name = "--" + name
                if hasattr(annotation.__metadata__[0], "default"):
                    arg_name = annotation.__metadata__[0].default or arg_name
                self.kwargs_renaming[name] = arg_name.lstrip("-")
                # Add argument
                parser.add_argument(
                    arg_name,
                    type=type_,
                    help=f"type: {type_.__name__}",
                    default=param.default,
                    action=action,  # type: ignore[arg-type]
                )
            self.registered_command = func
            return func

        return wrapper

    def __call__(self, *args, **kwargs):
        assert self.registered_command
        a = parser.parse_args()
        ka = dict(a._get_kwargs())
        for real_key, pseudo_key in self.kwargs_renaming.items():
            ka[real_key] = ka.pop(pseudo_key)
        command = self.registered_command
        return command(**ka)


class OptionInfo:
    def __init__(self, default):
        self.default = default


def Option(*args, **kwargs):
    default = args[0] if args else None
    return OptionInfo(default)
