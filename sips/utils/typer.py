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
                if issubclass(type(annotation.__origin__), type):
                    type_ = annotation.__origin__
                elif annotation.__origin__._name == "Optional":
                    type_ = annotation.__origin__.__args__[0]
                else:
                    raise NotImplementedError(f"Unknown {annotation=}")
                # Set argument name and help
                help_ = f"type: {type_.__name__}"
                if param.default is inspect._empty:
                    arg_name = name
                else:
                    arg_name = "--" + name
                if isinstance(annotation.__metadata__[0], OptionInfo):
                    option_info = annotation.__metadata__[0]
                    arg_name = option_info.default or arg_name
                    help_ = option_info.help or help_
                self.kwargs_renaming[name] = arg_name.lstrip("-")
                # Add argument
                parser.add_argument(
                    arg_name,
                    type=type_,
                    help=help_,
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
    def __init__(self, default, help):
        self.default = default
        self.help = help


def Option(default: str | None = None, *, help: str | None = None, **kwargs):
    return OptionInfo(default, help)
