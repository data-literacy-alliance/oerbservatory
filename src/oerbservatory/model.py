"""A data model for open educational resources."""

import datetime
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import numpy as np
import orcid_downloader
import pandas as pd
import pystow
import rdflib
import ssslm
from curies import Reference
from pydantic import UUID4, BaseModel, ByteSize, ConfigDict, Field
from pydantic_extra_types.language_code import ISO639_3, LanguageAlpha2
from rdflib import Literal, URIRef
from tqdm import tqdm

__all__ = [
    "Author",
    "EducationalResource",
    "Organization",
    "resolve_authors",
    "write_resources_jsonl",
    "write_resources_sentence_transformer",
    "write_resources_tfidf",
    "write_sqlite_fti",
]


class Author(BaseModel):
    """Represents an author."""

    name: str | None = None
    orcid: str | None = None

    def get_name(self) -> str | None:
        """Get the name from ORCID if possible, otherwise fall back to local."""
        if self.orcid and (name := orcid_downloader.get_name(self.orcid)) is not None:
            return name
        return self.name


class Organization(BaseModel):
    """Represents an organization."""

    name: str
    ror: str | None = None
    wikidata: str | None = None


type InternationalizedStr = dict[LanguageAlpha2, str]


class EducationalResource(BaseModel):
    """Represents an educatioanl resource."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    uuid: UUID4 = Field(default_factory=uuid4)
    reference: Reference | None = None

    platform: str = Field(
        ...,
        description="A key for the OER platform where this resource came from",
    )
    authors: list[Author | Organization] = Field(default_factory=list)
    license: str | URIRef | None = None
    external_uri: str | None | list[str] = None
    title: InternationalizedStr
    supporting_community: list[Organization] = Field(default_factory=list)
    recommending_community: list[Organization] = Field(default_factory=list)
    description: InternationalizedStr | None = None
    disciplines: list[URIRef] = Field(default_factory=list)

    keywords: list[InternationalizedStr] = Field(default_factory=list)
    date_published: datetime.datetime | datetime.date | None = None
    resource_types: list[URIRef] = Field(default_factory=list, description="Media types")
    media_types: list[URIRef] = Field(
        default_factory=list, description="somehow not the same as media types?"
    )
    difficulty_level: list[URIRef] = Field(default_factory=list)
    languages: list[ISO639_3] = Field(default_factory=list)
    audience: list[URIRef] = Field(default_factory=list)  # TODO add to RDF

    file_size: ByteSize | None = Field(None, description="file size, in bytes")
    file_formats: list[str] = Field(default_factory=list, examples=[["pdf"], ["mp4"]])
    xrefs: list[Reference] = Field(default_factory=list)
    logo: str | None = None
    version: str | None = None

    derived_from: str | None = Field(
        None,
        description="When deriving this OER object from an external OER resource, "
        "keep the external URI/ID",
    )

    @property
    def best_title(self) -> str:
        """Get the best title, prioritizing english, then german, then whatever."""
        for language_code in ["en", "de"]:
            if language_code in self.title:
                return self.title[LanguageAlpha2(language_code)]
        return self.title[min(self.title)]

    @staticmethod
    def _add(
        graph: rdflib.Graph,
        node: URIRef,
        predicate: URIRef,
        x: None | str | InternationalizedStr,
    ) -> None:
        if x is None:
            return None
        elif isinstance(x, str):
            graph.add((node, predicate, Literal(x)))
        elif isinstance(x, dict):
            for lang, text in x.items():
                graph.add((node, predicate, Literal(text, lang=lang)))
        else:
            raise TypeError


def write_resources_jsonl(resources: list[EducationalResource], path: Path) -> None:
    """Write resources as a JSONL file."""
    with path.open("w") as file:
        for resource in resources:
            line = resource.model_dump_json(
                exclude_none=True,
                exclude_defaults=True,
                exclude_unset=True,
            )
            file.write(line + "\n")


def _get_document(resource: EducationalResource) -> str:
    r = ""
    if resource.title:
        r += " ".join(resource.title.values())
    if resource.description:
        r += " ".join(resource.description.values())
    if resource.keywords:
        r += " ".join(v for keyword in resource.keywords for v in keyword.values())
    return r


def write_resources_tfidf(
    resources: list[EducationalResource],
    vectors_path: Path,
    similarities_path: Path,
    *,
    similarity_cutoff: float | None = None,
) -> None:
    """Create a vector index with TF-IDF."""
    from nltk.corpus import stopwords
    from pystow import ensure_nltk
    from sklearn.feature_extraction.text import TfidfVectorizer

    ensure_nltk()

    stop_words = stopwords.words(["english", "german"])

    corpus = [_get_document(resource) for resource in resources]
    vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True)

    vectors = vectorizer.fit_transform(corpus)

    _xxx(
        vectors=vectors.toarray(),
        resources=resources,
        vectors_path=vectors_path,
        similarities_path=similarities_path,
        cutoff=similarity_cutoff,
        columns=vectorizer.get_feature_names_out(),
    )


"""

## Document Featurization

By running `python -m dalia_ingest.sources.dalia`, two text indexes are
generated:

1. Using term frequency-inverse document frequency (TF-IDF)
2. Using [sentence transformers (SBERT)](https://sbert.net/)

Each method turns the combination of title, description, and keywords for an OER
into a single vector, which can then be compared to the vectors of all other
OERs via the cosine similarity. For each method, there's a file `-index.tsv`
that has the vectors and a `-similarities.tsv` file that makes a similarity
cutoff.

These can be used for downstream tasks like:

1. Clustering similar documents / finding similar documents
2. Training classifiers, e.g., for audience type or education level

"""


def write_resources_sentence_transformer(
    resources: list[EducationalResource],
    vectors_path: Path,
    similarity_path: Path,
    *,
    similarity_cutoff: float | None = None,
) -> None:
    """Create a vector index with :mod:`sentence_transformers`."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "distiluse-base-multilingual-cased",
        cache_folder=pystow.module("sentence-transformers").base.as_posix(),
    )
    corpus = [_get_document(resource) for resource in resources]
    vectors = model.encode(corpus, show_progress_bar=True)
    _xxx(
        vectors=vectors,
        resources=resources,
        vectors_path=vectors_path,
        similarities_path=similarity_path,
        cutoff=similarity_cutoff,
    )


def _xxx(
    vectors: np.ndarray,
    resources: list[EducationalResource],
    vectors_path: Path,
    similarities_path: Path,
    cutoff: float | None = None,
    columns: list[str] | None = None,
) -> None:
    from sklearn.metrics.pairwise import cosine_similarity

    index = [r.reference.curie if r.reference else str(r.uuid) for r in resources]

    df = pd.DataFrame(
        vectors,
        columns=columns,
        index=index,
    )
    df.index.name = "curie"
    df.to_csv(vectors_path, index=True, sep="\t")

    if cutoff is None:
        cutoff = 0.7

    similarity = cosine_similarity(vectors)
    high_similarity = []
    for i in range(len(resources)):
        for j in range(i):
            if i == j:
                continue
            sim = similarity[i][j]
            if sim > cutoff:
                high_similarity.append(
                    (
                        index[i],
                        resources[i].best_title,
                        index[j],
                        resources[j].best_title,
                        sim,
                    )
                )

    df2 = pd.DataFrame(high_similarity)
    df2.to_csv(similarities_path, sep="\t", index=False)


def write_sqlite_fti(resources: list[EducationalResource], path: Path) -> None:
    """Write a SQLite database with a full text index."""
    from dalia_dif.dif13.export.fti import _dif13_df_to_sqlite

    path.unlink(missing_ok=True)

    df = pd.DataFrame(
        [
            (
                resource.reference.identifier if resource.reference else str(resource.uuid),
                " ".join(resource.title.values()) if resource.title else "",
                " ".join(resource.description.values()) if resource.description else "",
                " ".join(v for keyword in resource.keywords for v in keyword.values())
                if resource.keywords
                else "",
            )
            for resource in resources
        ],
        columns=["uuid", "title", "description", "keywords"],
    )

    with closing(sqlite3.connect(path.as_posix())) as conn:
        _dif13_df_to_sqlite(df, conn)

        with closing(conn.cursor()) as cursor:
            # Test FTS query (e.g., search all fields for "python")
            # note that the bm25 weights
            query = """
                SELECT uuid, title, bm25(documents, 0.0, 5.0, 1.0, 0.5)
                FROM documents
                WHERE documents MATCH 'chem*'
                ORDER BY rank;
            """
            results = cursor.execute(query).fetchall()
            # Show results
            for row in results:
                tqdm.write(str(row))


def resolve_authors(
    author_names: list[str],
    *,
    ror_grounder: ssslm.Grounder,
) -> Sequence[Author | Organization]:
    """Get authors."""
    rv = []
    # wishing for better content in https://github.com/ElixirTeSS/TeSS/issues/1116
    for author_name in author_names:
        author_name = author_name.strip()
        if author_name.lower() in {"unknown", "unknown unknown"}:
            continue

        author: Author | Organization
        if "orcid:" in author_name:
            # this means it's like "Valipour Kahrood, Hossein (orcid: 0000-0003-4166-0382)"
            name, _, orcid = author_name.partition("(orcid:")
            name = name.strip()
            orcid = orcid.strip().strip(")").strip()
            author = Author(name=name, orcid=orcid)
        else:
            matches = orcid_downloader.ground_researcher(author_name)
            if len(matches) == 1:
                author = Author(name=matches[0].name, orcid=matches[0].identifier)
            else:
                matches = ror_grounder.get_matches(author_name)
                if len(matches) == 1:
                    author = Organization(name=matches[0].name, ror=matches[0].identifier)
                else:
                    author = Author(name=author_name)
        rv.append(author)

    return rv
