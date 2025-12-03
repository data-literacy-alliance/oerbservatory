"""Ingest TeSS.

See data dictionaries at https://github.com/ElixirTeSS/TeSS/tree/master/config/dictionaries.
"""

from collections import Counter
from collections.abc import Sequence
from functools import lru_cache

import click
import pyobo
import pystow
import ssslm
import tess_downloader
from curies import Reference
from dalia_dif.namespace import BIBO, HCRT, MODALIA
from rdflib import SDO, URIRef
from tabulate import tabulate
from tess_downloader import INSTANCES, DifficultyLevel, LearningMaterial, TeSSClient
from tqdm import tqdm

from oerbservatory.model import EN, Author, EducationalResource, Organization, resolve_authors
from oerbservatory.sources.utils import TESS_TO_LICENSE

__all__ = [
    "get_single_tess",
    "get_tess",
    "map_tess_oer",
]

TESS_LICENSE_DICTIONARY_URL = (
    "https://github.com/ElixirTeSS/TeSS/raw/refs/heads/master/config/dictionaries/licences.yml"
)
NO_MATERIALS = {"dresa", "explora"}

OERBSERVATORY_MODULE = pystow.module("oerbservatory", "inputs", "tess")
RESOURCE_TYPE_MAP: dict[str, URIRef | None] = {
    "video": SDO.VideoObject,
    "series of videos": SDO.VideoObject,
    "youtube video": SDO.VideoObject,
    #
    "computer software": SDO.SoftwareSourceCode,
    "coding": SDO.SoftwareSourceCode,
    "scripts": SDO.SoftwareSourceCode,
    "programming": SDO.SoftwareSourceCode,
    #
    "poster": SDO.Poster,
    #
    "workshop": MODALIA["Workshop"],
    "podcast": SDO.PodcastEpisode,
    #
    "slidedeck": HCRT["slide"],
    "slideshow": HCRT["slide"],
    "slides": HCRT["slide"],
    "slide deck / presentation": HCRT["slide"],
    "slides / presentation": HCRT["slide"],
    "slideck/ presentation": HCRT["slide"],
    "presentation": HCRT["slide"],
    #
    "jupyter notebooks": MODALIA["CodeNotebook"],
    "jupyter notebook": MODALIA["CodeNotebook"],
    #
    "blog post": None,
    #
    "training materials": None,
    "examples": None,
    "documentation": None,
    "bioinformatics": None,
    "hands-on tutorial": None,
    "learning pathway": None,
    "tutorials": None,
    "handbook": None,
    "case studies": None,
    "implementation guidelines": None,
    "additional reading": None,
    "didactic activities": None,
    "mock data": None,
    "how-to guide": None,
    "online course": None,
    "online material": None,
    "education": None,
    "open educational resource": None,
    "tool": None,
    "toolkit": None,
    "e-learning + workshop": None,
    "pdf": None,
    "recording": None,
    "r shiny application": None,
    "free online course": None,
    "carpentries style curriculum": None,
    "training materials with mock data": None,
    "online modules": None,
    "hackathon": None,
    "vignette": None,
    "api reference": None,
    "educational materials": None,
    "exercise": None,
    "handout": None,
    "workflow": None,
    "installation instructions": None,
    "manual": None,
    "talk": None,
    "knowledgebase": None,
    "book": BIBO["book"],
    "notes": None,
    # Topics
    "computational biology": None,
    "computer science": None,
    "data science": None,
    "transcriptomics": None,
    "machine learning": None,
    # Databases
    "life sciences literature database": None,
    "life science literature database": None,
    "viralzone": None,
}
DIFFICULTY_LEVEL_MAP: dict[DifficultyLevel, URIRef | None] = {
    "advanced": MODALIA["Expert"],
    "beginner": MODALIA["Beginner"],
    "intermediate": MODALIA["Competent"],
    "notspecified": None,
}

unknown_resource_type: Counter[str] = Counter()


@lru_cache(1)
def get_key_to_license_uri() -> dict[str, URIRef]:
    """Get a dictionary from key to license URI based on TeSS's configuration."""
    rv = {}
    records = OERBSERVATORY_MODULE.ensure_yaml(url=TESS_LICENSE_DICTIONARY_URL)
    for key, record in records.items():
        if key in TESS_TO_LICENSE:
            rv[key] = TESS_TO_LICENSE[key]
        else:
            license_uri = record["reference"]
            if not license_uri.startswith("https://spdx.org") or not license_uri.endswith(".html"):
                raise ValueError(f"{key} missing extension: {license_uri}")
            rv[key] = URIRef(license_uri.removesuffix(".html"))
    return rv


def _get_resource_types(attributes: LearningMaterial) -> list[URIRef]:
    rv = []
    for resource_type in attributes.resource_type or []:
        resource_type = resource_type.lower().strip()
        nn = RESOURCE_TYPE_MAP.get(resource_type)
        if nn:
            rv.append(nn)
        else:
            if not unknown_resource_type[resource_type]:
                tqdm.write(click.style(f'"{resource_type}": None,', fg="red"))
            unknown_resource_type[resource_type] += 1
    return rv


def _get_difficulty_levels(oer: tess_downloader.LearningMaterial) -> list[URIRef] | None:
    if oer.difficulty_level is None or oer.difficulty_level == "notspecified":
        return None
    if difficulty_level := DIFFICULTY_LEVEL_MAP.get(oer.difficulty_level):
        return [difficulty_level]
    return None


def _get_authors(
    attributes: LearningMaterial, organization_grounder: ssslm.Grounder
) -> Sequence[Author | Organization]:
    return resolve_authors(attributes.authors or [], organization_grounder=organization_grounder)


def map_tess_oer(
    client: TeSSClient,
    material_wrapper: tess_downloader.LearningMaterialWrapper,
    *,
    organization_grounder: ssslm.Grounder,
) -> EducationalResource | None:
    """Map a TeSS OER to an OERbservatory OER."""
    material = material_wrapper.attributes
    doi = material.doi
    if doi:
        if not doi.strip() or " " in doi:
            doi = None
        else:
            doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/").strip()
            doi = f"https://doi.org/{doi}"

    reference = Reference(prefix=f"tess.{client.key}", identifier=str(material_wrapper.id))

    educational_resource = EducationalResource(
        reference=reference,
        external_uri=doi,
        title={EN: material.title.strip()},
        license=_get_license(material),
        description={EN: material.description.strip()},
        keywords=[{EN: kw.strip()} for kw in material.keywords or []],
        date_published=material.date_published,
        resource_types=_get_resource_types(material),
        difficulty_level=_get_difficulty_levels(material),
        authors=_get_authors(material, organization_grounder=organization_grounder),
    )
    return educational_resource


def get_single_tess(
    client: TeSSClient,
    *,
    organization_grounder: ssslm.Grounder | None = None,
) -> list[EducationalResource]:
    """Get a TeSS graph."""
    if organization_grounder is None:
        organization_grounder = pyobo.get_grounder("ror")

    try:
        materials = client.get_materials()
    except ValueError as e:
        tqdm.write(f"[tess.{client.key}] failed: {e}")
        raise

    rv = [
        educational_resource
        for material in materials
        if (
            educational_resource := map_tess_oer(
                client, material, organization_grounder=organization_grounder
            )
        )
    ]
    tqdm.write(f"[tess.{client.key}] created {len(rv):,} records")
    return rv


def _get_license(attributes: LearningMaterial) -> URIRef | str | None:
    if attributes.license is None or attributes.license == "notspecified":
        return None
    return get_key_to_license_uri()[attributes.license]


def get_tess(
    *,
    organization_grounder: ssslm.Grounder | None = None,
) -> list[EducationalResource]:
    """Get processed OERs from all known TeSS instances."""
    if organization_grounder is None:
        organization_grounder = pyobo.get_grounder("ror")
    resources = []
    for key in tqdm(INSTANCES, unit="instance", desc="[tess] processing"):
        client = TeSSClient(key=key)
        resources.extend(
            get_single_tess(
                client=client,
                organization_grounder=organization_grounder,
            )
        )
    return resources


@click.command()
def main() -> None:
    """Convert TeSS to DALIA."""
    organization_grounder = pyobo.get_grounder("ror")
    get_tess(organization_grounder=organization_grounder)
    click.echo(tabulate(unknown_resource_type.most_common()))


if __name__ == "__main__":
    main()
