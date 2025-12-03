"""CLI for OER sources."""

from collections.abc import Callable

import click
from dalia_ingest.model import (
    EducationalResource,
    write_resources_jsonl,
    write_resources_sentence_transformer,
    write_resources_tfidf,
    write_sqlite_fti,
)
from dalia_ingest.utils import ROOT
from tqdm import tqdm

__all__ = ["main"]

OUTPUT_DIR = ROOT.joinpath("output")
OUTPUT_DIR.mkdir(exist_ok=True)


@click.command()
def main() -> None:
    """Get OER sources."""
    from dalia_ingest.sources.dalia import get_dalia
    from dalia_ingest.sources.oerhub import get_oerhub
    from dalia_ingest.sources.oersi import get_oersi
    from dalia_ingest.sources.tess import get_tess

    functions: list[Callable[[], list[EducationalResource]]] = [
        get_oerhub,
        get_oersi,
        get_tess,
        get_dalia,
    ]
    resources: list[EducationalResource] = []
    source_iterator = tqdm(functions, desc="OER source", leave=False)
    for f in source_iterator:
        key = f.__name__.removeprefix("get_")
        source_iterator.set_description(key)
        specific = f()
        if not specific:
            tqdm.write(click.style(f"no resources found for {key}", fg="red"))
            continue

        d = OUTPUT_DIR.joinpath(key)
        d.mkdir(exist_ok=True)
        d.joinpath(key)

        write_resources_jsonl(resources, d.joinpath(f"{key}.jsonl"))

        if key == "dalia":
            write_resources_tfidf(
                resources,
                d.joinpath(f"{key}-tfidf-index.tsv"),
                d.joinpath(f"{key}-tfidf-similarities.tsv"),
            )
            write_resources_sentence_transformer(
                resources,
                d.joinpath(f"{key}-transformers-index.tsv"),
                d.joinpath(f"{key}-tranformers-similarities.tsv"),
            )
            write_sqlite_fti(resources, d.joinpath(f"{key}-sqlite-full-text-index.db"))

        resources.extend(specific)

    click.echo(f"got {len(resources):,} resources")


if __name__ == "__main__":
    main()
