"""Parse DALIA curation sheets."""

import csv
import re
from pathlib import Path

import click
from curies import Reference
from dalia_dif.dif13 import (
    AuthorDIF13,
    EducationalResourceDIF13,
    OrganizationDIF13,
    parse_dif13_row,
)
from dalia_ingest.model import (
    OUTPUT_DIR,
    Author,
    EducationalResource,
    Organization,
    write_resources_jsonl,
    write_resources_sentence_transformer,
    write_resources_tfidf,
    write_sqlite_fti,
)
from dalia_ingest.utils import DALIA_MODULE, get_dif13_paths
from pydantic import ByteSize
from tqdm import tqdm

__all__ = [
    "get_dalia",
]


D = OUTPUT_DIR.joinpath("dalia")
DALIA_PROCESSED_PATH = D.joinpath("dalia.jsonl")
DALIA_TTL_PATH = DALIA_MODULE.join(name="dalia.ttl")
DALIA_TFIDF_INDEX_PATH = D.joinpath("dalia-tfidf-index.tsv")
DALIA_TFIDF_SIM_PATH = D.joinpath("dalia-tfidf-similarities.tsv")
DALIA_TRANSFORMERS_INDEX_PATH = D.joinpath("dalia-transformers-index.tsv")
DALIA_TRANSFORMERS_SIM_PATH = D.joinpath("dalia-transformers-similarities.tsv")
DALIA_SQLITE_FTS_PATH = D.joinpath("dalia-fts-sqlite.db")

ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}(\d|X)$")

ORCID_URI_PREFIX = "https://orcid.org/"
ROR_URI_PREFIX = "https://ror.org/"
WIKIDATA_URI_PREFIX = "http://www.wikidata.org/entity/"
RE = re.compile(r"^(?P<name>.*)\s\((?P<relation>S|R|SR|RS)\)$")


def _log(path: Path, line: int, text: str) -> None:
    tqdm.write(f"[{path.name} line:{line}] {text}")


def parse(path: str | Path) -> list[EducationalResource]:
    """Parse DALIA records."""
    path = Path(path).expanduser().resolve()
    with path.open(newline="") as csvfile:
        return [
            oer
            for idx, record in enumerate(csv.DictReader(csvfile), start=2)
            if (oer := _omni_process_row(path, idx, record)) is not None
        ]


def _convert(e: EducationalResourceDIF13) -> EducationalResource | None:
    rv = EducationalResource(
        platform="dalia",
        reference=Reference(prefix="dalia.oer", identifier=str(e.uuid)),
        external_uri=e.links,
        title={"en": e.title},
        description={"en": e.description},
        keywords=[{"en": keyword} for keyword in e.keywords],
        authors=[_process_author(a) for a in e.authors],
        difficulty_level=e.proficiency_levels,
        languages=e.languages,
        license=e.license,
        file_formats=e.file_formats,
        date_published=e.publication_date,
        version=e.version,
        audience=e.target_groups,
        file_size=_process_size(e.file_size),
        resource_types=e.learning_resource_types,
        media_types=e.media_types,
        disciplines=e.disciplines,
    )
    return rv


def _process_size(x: str | None) -> ByteSize | None:
    if x is None:
        return None
    if not x.endswith(" MB"):
        raise ValueError
    return ByteSize(int(float(x.removesuffix(" MB")) * 1_000_000))


def _process_author(e: AuthorDIF13 | OrganizationDIF13) -> Author | Organization:
    match e:
        case AuthorDIF13():
            return Author(
                name=e.name,
                orcid=e.orcid,
            )
        case OrganizationDIF13():
            return Organization(
                name=e.name,
                ror=e.ror,
                wikidata=e.wikidata,
            )
        case _:
            raise TypeError


def _omni_process_row(path: Path, idx: int, row: dict[str, str]) -> EducationalResource | None:
    """Convert a row in a DALIA curation file to a resource, or return none if unable."""
    ed13 = parse_dif13_row(path, idx, row, future=True)
    if ed13 is None:
        return None
    return _convert(ed13)


def get_dalia() -> list[EducationalResource]:
    """Get processed OERs from DALIA."""
    return [resource for path in get_dif13_paths() for resource in parse(path)]


@click.command()
@click.option("--transformers", is_flag=True)
def main(transformers: bool) -> None:
    """Process DALIA curation sheets."""
    resources = get_dalia()
    write_resources_jsonl(resources, DALIA_PROCESSED_PATH)

    write_sqlite_fti(resources, DALIA_SQLITE_FTS_PATH)

    if transformers:
        write_resources_tfidf(resources, DALIA_TFIDF_INDEX_PATH, DALIA_TFIDF_SIM_PATH)
        write_resources_sentence_transformer(
            resources,
            DALIA_TRANSFORMERS_INDEX_PATH,
            DALIA_TRANSFORMERS_SIM_PATH,
        )


if __name__ == "__main__":
    main()
