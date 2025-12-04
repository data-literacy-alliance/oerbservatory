"""
Galaxy Training Network (GTN).

Training materials are in this repository: https://github.com/galaxyproject/training-material. Many
have markdown files that have front-matter that describes their metadata, such as
https://github.com/galaxyproject/training-material/blob/main/topics/data-science/tutorials/cli-bashcrawl/tutorial.md

There's also an API that takes care of parsing this and exposing it through the topics endpoint
https://galaxyproject.github.io/training-material/api/.

"""
from pathlib import Path
from typing import Any

import pystow
import requests
from tqdm import tqdm
from collections import Counter

from dalia_dif.namespace import MODALIA
from oerbservatory.model import EducationalResource, write_resources_jsonl

__all__ = [
    "get_gtn",
]

MODULE = pystow.module("oerbservatory", "sources", "gtn")

missing_field_counter = Counter()

LEVEL_MAP = {
    "Advanced": MODALIA["Expert"],
    "Beginner": MODALIA["Beginner"],
    "Intermediate": MODALIA["Competent"],
    "Introductory": MODALIA["Novice"],
}


def get_gtn(refresh: bool = False) -> list[EducationalResource]:
    """Get learning materials from GTN."""
    topics = MODULE.ensure_json(
        url="https://training.galaxyproject.org/training-material/api/topics.json", force=refresh
    )
    for topic in tqdm(topics, desc="Getting GTN topics"):
        if topic == "admin":
            continue
        url = f"https://training.galaxyproject.org/training-material/api/topics/{topic}.json"
        res_json = MODULE.ensure_json(url=url)
        # res = requests.get(url, timeout=30)
        # res.raise_for_status()
        # res_json = res.json()
        for material in tqdm(res_json["materials"], leave=False, desc=f"GTN: {topic}"):
            if educational_resource := _process_material(topic, material):
                yield educational_resource


def _process_material(topic, record: dict[str, Any]) -> EducationalResource | None:
    topic_name = record.pop("tutorial_name")
    url = f"https://github.com/galaxyproject/training-material/raw/refs/heads/main/topics/{topic}/tutorials/{topic_name}/tutorial.md"

    try:
        if False:
            res = requests.get(url, timeout=5)
            text = res.text
        else:
            path = MODULE.ensure(url=url, name=f"{topic}-{topic_name}-tutorial.md")
            text = path.read_text()
    except pystow.utils.DownloadError:
        tqdm.write(f"[{topic}-{topic_name}] was not able to download {url}")
        description = ""
    else:
        rest = text.split('---', 3)[2].strip()
        index = rest.find("\n")
        description = rest[:index]

    if questions := record.pop("questions", []):
        fmt_text = "\n".join(f"- {question}" for question in questions)
        description += f"\n\nThis tutorial covers the following questions:\n{fmt_text}\n"

    if key_points := record.pop("key_points", []):
        fmt_text = "\n".join(f"- {key_point}" for key_point in key_points)
        description += f"\n\nThis tutorial covers the key points:\n{fmt_text}\n"

    rv = EducationalResource(
        title={"en": record.pop("title")},
        description={"en": description.strip()},
        learning_objectives="\n-".join(objectives) if (objectives := record.pop("objectives", [])) else None,
        difficulty_level=LEVEL_MAP[level] if (level := record.get('level')) else None,
    )
    for key in record:
        missing_field_counter[key] += 1
    return rv


if __name__ == '__main__':
    write_resources_jsonl(get_gtn(), Path.home().joinpath("gtn.jsonl"))
