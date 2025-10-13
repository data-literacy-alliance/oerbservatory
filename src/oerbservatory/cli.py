"""Command line interface for :mod:`oerbservatory`."""

import click

__all__ = [
    "main",
]


@click.command()
def main() -> None:
    """CLI for oerbservatory."""


if __name__ == "__main__":
    main()
