"""Ingest OERhub."""

import json
from collections import Counter
from typing import Any, cast

import click
import pyobo
import pystow
import requests
import ssslm
from curies import Reference
from dalia_dif.namespace import SPDX_LICENSE
from dalia_dif.utils import cleanup_languages
from pydantic_extra_types.language_code import LanguageAlpha2
from rdflib import SDO, URIRef
from tabulate import tabulate
from tqdm import tqdm

from oerbservatory.model import (
    EducationalResource,
    InternationalizedStr,
    resolve_authors,
    write_resources_jsonl,
)
from oerbservatory.sources.utils import OUTPUT_DIR

__all__ = [
    "get_oerhub",
    "get_oerhub_raw",
]

OERHUB_MODULE = pystow.module("oerbservatory", "sources", "oerhub")
OERHUB_RAW_PATH = OERHUB_MODULE.join(name="oerhub-raw.json")
OERHUB_PROCESSED_PATH = OERHUB_MODULE.join(name="oerhub.jsonl")
OERHUB_TTL_PATH = OUTPUT_DIR.joinpath("oerhub.ttl")


def get_oerhub_raw(*, force: bool = False) -> dict[str, Any]:
    """Get OERhub data."""
    if OERHUB_RAW_PATH.is_file() and not force:
        return cast(dict[str, Any], json.loads(OERHUB_RAW_PATH.read_text()))

    url = "https://oerhub.at/search"
    # there were 3143 on June 20, 2025
    params = {"query": "*", "page": 0, "size": 10000}
    res = requests.post(url, json=params, timeout=60)
    data = res.json()
    with OERHUB_RAW_PATH.open("w") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    return cast(dict[str, Any], data)


LICENSES: dict[str, URIRef] = {
    "CC-BY-4.0": SPDX_LICENSE["CC-BY-4.0"],
    "CC-BY-3.0-AT": SPDX_LICENSE["CC-BY-3.0"],
    "CC-BY-SA-4.0": SPDX_LICENSE["CC-BY-SA-4.0"],
    "CC-BY-ND-4.0": SPDX_LICENSE["CC-BY-ND-4.0"],
    "CC-BY-SA-3.0-AT": SPDX_LICENSE["CC-BY-SA-3.0"],
    "CC-BY-NC-4.0": SPDX_LICENSE["CC-BY-NC-4.0"],
    "CC-BY-NC-ND-4.0": SPDX_LICENSE["CC-BY-NC-ND-4.0"],
    "CC-BY-NC-SA-4.0": SPDX_LICENSE["CC-BY-NC-SA-4.0"],
    "CC-BY-SA-2.0": SPDX_LICENSE["CC-BY-SA-2.0"],
}

RESOURCE_TYPES = {
    "Document": SDO.DigitalDocument,
    "Video": SDO.VideoObject,
    "Picture": SDO.Photograph,
    "unknown": None,
    "Miscellaneous": None,
    "iMooX": None,
}


def get_oerhub(*, organization_grounder: ssslm.Grounder | None = None) -> list[EducationalResource]:  # noqa:C901
    """Get processed OERs from OERhub."""
    data = get_oerhub_raw()
    hits = data["data"]["hits"]["hits"]

    if organization_grounder is None:
        organization_grounder = pyobo.get_grounder("ror")

    mime_type_counter: Counter[str] = Counter()
    filetype_counter: Counter[str] = Counter()

    resources: list[EducationalResource] = []
    key_counter: Counter[str] = Counter()
    key_examples = {}
    count = 0
    for record in tqdm(hits, unit="OER", unit_scale=True, desc="Processing OERhub"):
        source = record["_source"]
        general = source.pop("general")
        technical = source.pop("technical")

        # only applies to video
        technical.pop("duration", None)
        # not sure what this is
        source.pop("oea_valid", None)
        # unneeded metadata on when ingestion happened
        source.pop("oea_ingest", None)
        # can be reconstructed with references
        source.pop("oea_object_direct_link", None)

        title: InternationalizedStr | None
        title_1: list[InternationalizedStr] = general.pop("title")
        title_2: str | None = source.pop("oea_title", None)
        title_3: InternationalizedStr | None = source.pop("oea_title_ml", None)  # this is a dict
        if title_3:
            title = _clean_d(title_3)
        elif title_1:
            title = _clean_d(title_1[0])
        elif title_2:
            title = {LanguageAlpha2("de"): title_2}
        else:
            continue

        license_classification = source.pop("oea_classification_02").strip()
        rights = source.pop("rights")
        if license_classification:
            license = LICENSES[license_classification]
        else:
            tqdm.write(f"no license classification detected. Rights: {rights}")
            license = None  # TODO processs rights?

        keywords: list[InternationalizedStr] = [
            {
                LanguageAlpha2("en"): x["name_en"],
                LanguageAlpha2("de"): x["name_de"],
            }
            for x in source.pop("oea_classification_01")
        ]

        thumbnail_url: str | None = source.pop("oea_thumbnail_url", None)
        # there's also a description available here
        thumbnail_url_2: str | None = technical.pop("thumbnail", {}).get("url")

        descriptions = general.pop("description", [])
        description = _clean_d(descriptions[0]) if descriptions else None

        media_types = []
        resource_type = source.pop("oea_classification_00")
        if rr_ := RESOURCE_TYPES[resource_type]:
            media_types.append(rr_)

        direct_link = source.pop("oea_object_direct_link", None)
        # oea_classification_04 is also resource type?

        mime_type = technical.pop("format")  # unused
        mime_type_counter[mime_type] += 1
        file_format = source.pop("oea_classification_05")
        if file_format == "unknown":
            file_format = None
        filetype_counter[file_format] += 1

        source.pop("oea_classification_06")
        languages = cleanup_languages(general.pop("language", []))

        r = EducationalResource(
            platform="oerhub",
            external_uri=direct_link,
            title=title,
            license=license,
            keywords=keywords,
            description=description,
            languages=languages,
            authors=resolve_authors(
                source.pop("oea_authors", []), organization_grounder=organization_grounder
            ),
            xrefs=[
                # TODO register all to bioregistry!
                Reference(prefix=x["catalog"], identifier=x["entry"])
                for x in general.pop("identifiers")
            ],
            logo=thumbnail_url or thumbnail_url_2,
            date_published=source.pop("oea_classification_03"),
            resource_types=media_types,
            file_size=technical.pop("size"),
            file_formats=[file_format] if file_format else [],
        )

        for d in [source, general, technical]:
            for key, value in d.items():
                if value:
                    key_counter[key] += 1
                    if key not in key_examples:
                        key_examples[key] = value

        resources.append(r)
        count += 1

    tqdm.write(f"[oerhub] got {count:,} records")

    _echo_counter(mime_type_counter, title="Formats")
    _echo_counter(filetype_counter, title="Filetype")

    rows = [(k, v, key_examples[k]) for k, v in sorted(key_counter.items())]
    tqdm.write(tabulate(rows, headers=["key", "count", "example"]))
    return resources


def _echo_counter(c: Counter[str], title: str | None = None) -> None:
    if title:
        tqdm.write(title)
    tqdm.write(tabulate(c.most_common(), headers=["key", "count"]) + "\n\n")


def _clean_d(d: InternationalizedStr | None) -> InternationalizedStr | None:
    if d is None:
        return None
    if "en" in d and "en_us_wp" in d:
        del d[LanguageAlpha2("en_us_wp")]
        return d
    return {LanguageAlpha2("en") if k == "en_us_wp" else k: v for k, v in d.items()}


@click.command()
def main() -> None:
    """Process OERhub."""
    resources = get_oerhub()
    write_resources_jsonl(resources, OERHUB_PROCESSED_PATH)


if __name__ == "__main__":
    main()
