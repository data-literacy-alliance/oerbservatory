"""CLI for OER sources."""

import time
from collections.abc import Callable

import click
from tqdm import tqdm

from oerbservatory.model import (
    EducationalResource,
    write_resources_jsonl,
    write_resources_sentence_transformer,
    write_resources_tfidf,
    write_sqlite_fti,
)
from oerbservatory.sources.utils import OUTPUT_DIR

__all__ = ["main"]


@click.command()
def main() -> None:
    """Get OER sources."""
    from oerbservatory.sources.dalia import get_dalia
    from oerbservatory.sources.gtn import get_gtn
    from oerbservatory.sources.oerhub import get_oerhub
    from oerbservatory.sources.tess import get_tess

    source_getters: list[Callable[[], list[EducationalResource]]] = [
        get_tess,
        get_dalia,
        get_oerhub,
        get_gtn,
        # get_oersi,
    ]
    concat_sources: list[EducationalResource] = []
    source_getters_it = tqdm(source_getters, desc="OER source", leave=False)
    for get_resources in source_getters_it:
        key = get_resources.__name__.removeprefix("get_")
        source_getters_it.set_description(key)
        resources = get_resources()
        if not resources:
            tqdm.write(click.style(f"no resources found for {key}", fg="red"))
            continue

        d = OUTPUT_DIR.joinpath(key)
        d.mkdir(exist_ok=True)

        tqdm.write(f"[{key}] outputting JSONL to {d}")
        write_resources_jsonl(resources, d.joinpath(f"{key}.jsonl"))

        tqdm.write(f"[{key}] calculating TF-IDF vectors")
        start = time.time()
        write_resources_tfidf(
            resources,
            d.joinpath(f"{key}-tfidf-index.tsv"),
            d.joinpath(f"{key}-tfidf-similarities.tsv"),
        )
        tqdm.write(f"[{key}] output TF-IDF vectors to {d} in {time.time() - start:.2f} seconds")

        tqdm.write(f"[{key}] calculating SBERT vectors")
        start = time.time()
        write_resources_sentence_transformer(
            resources,
            d.joinpath(f"{key}-transformers-index.tsv"),
            d.joinpath(f"{key}-tranformers-similarities.tsv"),
        )
        tqdm.write(f"[{key}] output SBERT vectors to {d} in {time.time() - start:.2f} seconds")

        if key == "dalia":
            write_sqlite_fti(resources, d.joinpath(f"{key}-sqlite-full-text-index.db"))

        concat_sources.extend(resources)

    click.echo(f"got {len(concat_sources):,} resources")

    write_resources_jsonl(concat_sources, OUTPUT_DIR.joinpath("all.jsonl"))

    tqdm.write("[ALL] calculating TF-IDF vectors")
    start = time.time()
    write_resources_tfidf(
        concat_sources,
        OUTPUT_DIR.joinpath("tfidf-index.tsv"),
        OUTPUT_DIR.joinpath("tfidf-similarities.tsv"),
    )
    tqdm.write(f"output TF-IDF vectors to {OUTPUT_DIR} in {time.time() - start:.2f} seconds")

    tqdm.write("calculating SBERT vectors")
    start = time.time()
    write_resources_sentence_transformer(
        concat_sources,
        OUTPUT_DIR.joinpath("transformers-index.tsv"),
        OUTPUT_DIR.joinpath("transformers-similarities.tsv"),
    )
    tqdm.write(f"output SBERT vectors to {OUTPUT_DIR} in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
