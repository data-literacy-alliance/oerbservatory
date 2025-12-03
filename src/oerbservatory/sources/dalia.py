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
from pydantic import ByteSize
from tqdm import tqdm

from oerbservatory.model import (
    EN,
    Author,
    EducationalResource,
    Organization,
    write_resources_jsonl,
    write_resources_sentence_transformer,
    write_resources_tfidf,
    write_sqlite_fti,
)
from oerbservatory.sources.utils import OUTPUT_DIR

__all__ = [
    "get_dalia",
    "map_dalia_oer",
]

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


def map_dalia_oer(dalia_oer: EducationalResourceDIF13) -> EducationalResource | None:
    """Map a DALIA OER to an OERbservatory OER."""
    rv = EducationalResource(
        reference=Reference(prefix="dalia.oer", identifier=str(dalia_oer.uuid)),
        external_uri=dalia_oer.links,
        title={EN: dalia_oer.title},
        description={EN: dalia_oer.description},
        keywords=[{EN: keyword} for keyword in dalia_oer.keywords],
        authors=[_process_author(a) for a in dalia_oer.authors],
        difficulty_level=dalia_oer.proficiency_levels,
        languages=dalia_oer.languages,
        license=dalia_oer.license,
        file_formats=dalia_oer.file_formats,
        date_published=dalia_oer.publication_date,
        version=dalia_oer.version,
        audience=dalia_oer.target_groups,
        file_size=_process_size(dalia_oer.file_size),
        resource_types=dalia_oer.learning_resource_types,
        media_types=dalia_oer.media_types,
        disciplines=dalia_oer.disciplines,
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
            return Author(name=e.name, orcid=e.orcid)
        case OrganizationDIF13():
            return Organization(name=e.name, ror=e.ror, wikidata=e.wikidata)
        case _:
            raise TypeError


def _omni_process_row(path: Path, idx: int, row: dict[str, str]) -> EducationalResource | None:
    """Convert a row in a DALIA curation file to a resource, or return none if unable."""
    ed13 = parse_dif13_row(path.name, idx, row, future=True)
    if ed13 is None:
        return None
    return map_dalia_oer(ed13)


def get_dif13_paths() -> list[Path]:
    """Get DALIA curation paths."""
    base = Path("/Users/cthoyt/dev/dalia-curation/curation")
    return list(base.glob("*.tsv"))


def get_dalia() -> list[EducationalResource]:
    """Get processed OERs from DALIA."""
    return [resource for path in get_dif13_paths() for resource in parse(path)]


@click.command()
@click.option("--transformers", is_flag=True)
def main(transformers: bool) -> None:
    """Process DALIA curation sheets."""
    resources = get_dalia()
    dire = OUTPUT_DIR.joinpath("dalia")
    write_resources_jsonl(resources, dire.joinpath("dalia.jsonl"))

    write_sqlite_fti(resources, dire.joinpath("dalia-fts-sqlite.db"))

    if transformers:
        write_resources_tfidf(
            resources,
            dire.joinpath("dalia-tfidf-index.tsv"),
            dire.joinpath("dalia-tfidf-similarities.tsv"),
        )
        write_resources_sentence_transformer(
            resources,
            dire.joinpath("dalia-transformers-index.tsv"),
            dire.joinpath("dalia-transformers-similarities.tsv"),
        )


if __name__ == "__main__":
    main()
