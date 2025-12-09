"""
Galaxy Training Network (GTN).

Training materials are in this repository: https://github.com/galaxyproject/training-material. Many
have markdown files that have front-matter that describes their metadata, such as
https://github.com/galaxyproject/training-material/blob/main/topics/data-science/tutorials/cli-bashcrawl/tutorial.md

There's also an API that takes care of parsing this and exposing it through the topics endpoint
https://galaxyproject.github.io/training-material/api/.

"""

import textwrap
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import click
import dateutil.parser
import pystow
import requests
from curies import Reference
from dalia_dif.namespace import HCRT, MODALIA, modalia
from tabulate import tabulate
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from oerbservatory.model import EducationalResource, write_resources_jsonl

__all__ = [
    "get_gtn",
]

MODULE = pystow.module("oerbservatory", "sources", "gtn")
SITE_BASE = "https://training.galaxyproject.org/training-material"

missing_field_counter: Counter[str] = Counter()
examples = {}
LEVEL_MAP = {
    "Advanced": MODALIA["Expert"],
    "Beginner": MODALIA["Beginner"],
    "Intermediate": MODALIA["Competent"],
    "Introductory": MODALIA["Novice"],
}


def get_gtn(refresh: bool = False) -> list[EducationalResource]:
    """Get learning materials from GTN."""
    return list(iter_gtn(refresh=refresh))


def iter_gtn(refresh: bool = False) -> Iterable[EducationalResource]:
    """Iterate over learning materials from GTN."""
    topics = MODULE.ensure_json(
        url="https://training.galaxyproject.org/training-material/api/topics.json", force=refresh
    )
    for topic in tqdm(topics, desc="Getting GTN topics"):
        if topic == "admin":
            continue
        url = f"https://training.galaxyproject.org/training-material/api/topics/{topic}.json"
        if False:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            res_json = res.json()
        else:
            res_json = MODULE.ensure_json(url=url)
        for material in tqdm(res_json["materials"], leave=False, desc=f"GTN: {topic}"):
            if educational_resource := _process_material(topic, material):
                yield educational_resource


def _process_material(  # noqa:C901
    topic: str,
    record: dict[str, Any],
) -> EducationalResource | None:
    topic_name = record.pop("tutorial_name")
    url = f"https://github.com/galaxyproject/training-material/raw/refs/heads/main/topics/{topic}/tutorials/{topic_name}/tutorial.md"

    for key in [
        "js_requirements",
        "layout",
        "priority",
        "admin_install",
        "admin_install_yaml",
        "tours",
        # this sometimes points to the dataset(s) used in the tutorial, but
        # isn't the educational material itself
        "zenodo_link",
    ]:
        record.pop(key, None)

    lang: str = record.pop("lang", "en")

    match record.pop("type"):
        case "tutorial":
            resource_type = modalia.Tutorial
        case "slides":
            resource_type = HCRT["slide"]
        case x:
            raise ValueError(f"unhandled type: {x}")

    try:
        if False:
            res = requests.get(url, timeout=5)
            text = res.text
        else:
            with logging_redirect_tqdm():
                path = MODULE.ensure(url=url, name=f"{topic}-{topic_name}-tutorial.md")
            text = path.read_text()
    except pystow.utils.DownloadError:
        tqdm.write(f"[{topic}-{topic_name}] was not able to download {url}")
        description = ""
    else:
        rest = text.split("---", 3)[2].strip()
        index = rest.find("\n")
        description = rest[:index]

    if questions := record.pop("questions", []):
        fmt_text = "\n".join(f"- {question}" for question in questions)
        description += f"\n\nThis tutorial covers the following questions:\n{fmt_text}\n"

    if key_points := record.pop("key_points", []):
        fmt_text = "\n".join(f"- {key_point}" for key_point in key_points)
        description += f"\n\nThis tutorial covers the key points:\n{fmt_text}\n"

    if record.pop("draft", False):
        status = "Draft"
    else:
        status = "Active"

    xrefs = [
        Reference(prefix="edam", identifier=edam_id) for edam_id in record.pop("edam_ontology", [])
    ] or None

    keywords = []
    for tag in record.pop("tags", None) or []:
        keywords.append({lang: tag})
    if subtopic := record.pop("subtopic", None):
        keywords.append({lang: subtopic})

    rv = EducationalResource(
        reference=Reference(prefix="gtn", identifier=record.pop("short_id")),
        title={lang: record.pop("title")},
        description={lang: description.strip()},
        external_uri=f"{SITE_BASE}{record.pop('url')}",
        learning_objectives="\n-".join(objectives)
        if (objectives := record.pop("objectives", []))
        else None,
        difficulty_level=LEVEL_MAP[level] if (level := record.pop("level", None)) else None,
        modified=dateutil.parser.parse(modified) if (modified := record.pop("mod_date")) else None,
        published=dateutil.parser.parse(published)
        if (published := record.pop("pub_date"))
        else None,
        version=str(version) if (version := record.pop("version", None)) else None,
        license=record.pop("license", None),
        status=status,
        logo=record.pop("logo", None),
        resource_types=[resource_type] if resource_type else None,
        xrefs=xrefs,
        keywords=keywords or None,
    )
    for key in record:
        missing_field_counter[key] += 1
        if key not in examples and record[key]:
            examples[key] = record[key], topic, topic_name
    return rv


@click.command()
def main() -> None:
    """Test processing GTN and make a tabular summary of unhandled fields."""
    resources = list(iter_gtn())
    click.echo(
        tabulate(
            [
                (
                    key,
                    count,
                    textwrap.shorten(str(examples[key][0]), 100),
                    examples[key][1],
                    examples[key][2],
                )
                for key, count in missing_field_counter.most_common()
            ]
        )
    )
    write_resources_jsonl(resources, Path.home().joinpath("gtn.jsonl"))


if __name__ == "__main__":
    main()
