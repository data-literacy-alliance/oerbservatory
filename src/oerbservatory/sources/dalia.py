"""Parse DALIA curation sheets."""

import csv
import re
from collections.abc import Collection
from pathlib import Path

import click
from curies import Reference
from dalia_dif.dif13 import (
    AuthorDIF13,
    EducationalResourceDIF13,
    OrganizationDIF13,
    parse_dif13_row,
)
from dalia_dif.dif13.picklists import PROFICIENCY_TO_ORDER
from pydantic import ByteSize
from pydantic_extra_types.language_code import _index_by_alpha3
from rdflib import URIRef
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
from oerbservatory.sources.utils import OUTPUT_DIR, UNSPECIFIED_OR_PROPRIETARY

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


def _get_minimum_proficiency_level(pl: Collection[URIRef] | None) -> URIRef | None:
    if not pl:
        return None
    return min(pl, key=PROFICIENCY_TO_ORDER.__getitem__)


REMAINING_LICENSES = {
    URIRef("https://purl.org/ontology/modalia#ProprietaryLicense"): UNSPECIFIED_OR_PROPRIETARY,
}


def _process_license(license_uriref: URIRef | None) -> Reference | None | URIRef:
    if license_uriref is None:
        return None
    if license_uriref in REMAINING_LICENSES:
        return REMAINING_LICENSES[license_uriref]
    if license_uriref.startswith("http://spdx.org/licenses/"):
        return Reference(
            prefix="spdx", identifier=str(license_uriref).removeprefix("http://spdx.org/licenses/")
        )
    raise ValueError(f"unhandled license: {license_uriref}")


def map_dalia_oer(dalia_oer: EducationalResourceDIF13) -> EducationalResource | None:
    """Map a DALIA OER to an OERbservatory OER."""
    languages = dalia_oer.languages
    if languages:
        language_alpha2 = _index_by_alpha3()[languages[0]].alpha2
    else:
        language_alpha2 = EN

    external_uri, *external_uri_extras = dalia_oer.links

    rv = EducationalResource(
        reference=Reference(prefix="dalia.oer", identifier=str(dalia_oer.uuid)),
        external_uri=external_uri,
        external_uri_extras=external_uri_extras or None,
        title={language_alpha2: dalia_oer.title},
        description={language_alpha2: dalia_oer.description} if dalia_oer.description else None,
        keywords=[{language_alpha2: keyword} for keyword in dalia_oer.keywords],
        authors=[_process_author(a) for a in dalia_oer.authors],
        difficulty_level=_get_minimum_proficiency_level(dalia_oer.proficiency_levels),
        languages=languages,
        license=_process_license(dalia_oer.license),
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
            return Author(
                name=e.name,
                orcid=e.orcid.removeprefix("https://orcid.org/") if e.orcid is not None else None,
            )
        case OrganizationDIF13():
            return Organization(
                name=e.name,
                ror=e.ror.removeprefix("https://ror.org/") if e.ror is not None else None,
                wikidata=e.wikidata,
            )
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
    return list(base.glob("*.csv"))


def get_dalia() -> list[EducationalResource]:
    """Get processed OERs from DALIA."""
    return [resource for path in get_dif13_paths() for resource in parse(path)]


@click.command()
@click.option("--transformers", is_flag=True)
def main(transformers: bool) -> None:
    """Process DALIA curation sheets."""
    resources = get_dalia()
    dire = OUTPUT_DIR.joinpath("dalia")
    dire.mkdir(exist_ok=True, parents=True)
    write_resources_jsonl(resources, dire.joinpath("dalia.jsonl"))

    return
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
