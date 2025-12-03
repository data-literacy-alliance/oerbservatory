"""Process OERSI."""

import json
from collections.abc import Iterable
from typing import Any

import click
from dalia_ingest.model import (
    EducationalResource,
    write_resources_jsonl,
)
from dalia_ingest.utils import DALIA_MODULE
from pydantic_extra_types.language_code import _index_by_alpha2
from rdflib import URIRef
from tqdm import tqdm

__all__ = [
    "get_oersi",
    "get_oersi_raw",
]

URL = "https://oersi.org/dumps/oer_data.ndjson.gz"
OERSI_TTL_PATH = DALIA_MODULE.join(name="oersi.ttl")
OERSI_JSON_PATH = DALIA_MODULE.join(name="oersi.json")


def get_oersi_raw(*, force: bool = False) -> Iterable[dict[str, Any]]:
    """Get OERSI data."""
    with DALIA_MODULE.ensure_open_gz(url=URL, force=force) as file:
        for line in file:
            yield json.loads(line)


def get_oersi(*, force: bool = False) -> list[EducationalResource]:
    """Get processed records from OERSI."""
    return [_process(record) for record in tqdm(get_oersi_raw(force=force), unit_scale=True)]


def _process(record: dict[str, Any]) -> EducationalResource:
    skips = ["@context", "conditionsOfAccess"]
    for skip in skips:
        record.pop(skip, None)

    audiences = [
        # TODO standardize
        URIRef(a["id"])  # like http://purl.org/dcx/lrmi-vocabs/educationalAudienceRole/teacher
        for a in record.pop("audience", [])
    ]
    disciplines = [
        # TODO standardize
        URIRef(a["id"])  # like https://w3id.org/kim/hochschulfaechersystematik/n44
        for a in record.pop("about", [])
    ]
    description = record.pop("description", None)
    languages = [_index_by_alpha2()[language].alpha3 for language in record.pop("inLanguage", [])]
    resource_types = [
        # TODO standardize
        URIRef(t["id"])  # like https://w3id.org/kim/hcrt/textbook
        for t in record.pop("learningResourceType", [])
    ]

    # TODO type
    # TODO mainEntityOfPage
    # TODO publisher

    date_published = record.pop("datePublished", None)
    record.pop("license", {"id": None})["id"]  # TODO process to spdx
    name = record.pop("name")
    uri = record.pop("id")

    return EducationalResource(
        platform="oersi",
        external_uri=uri,
        title={"en": name},
        description={"en": description} if description else None,
        date_published=date_published,
        audience=audiences,
        resource_types=resource_types,
        disciplines=disciplines,
        languages=languages,
    )


@click.command()
def main() -> None:
    """Process OERSI."""
    resources = get_oersi()
    write_resources_jsonl(resources, OERSI_JSON_PATH)


if __name__ == "__main__":
    main()
