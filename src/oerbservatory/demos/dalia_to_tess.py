"""Demonstrate converting DALIA DIF v1.3 to TeSS."""

import json

from dalia_dif.dif13 import EducationalResourceDIF13, read_dif13
from dalia_dif.dif13.rdf import get_discipline_label
from tess_downloader import LearningMaterial, TeSSClient, Topic
from tess_downloader.api import PostLearningMaterial


def main() -> None:
    """Demonstrate converting DALIA DIF v1.3 to TeSS."""
    url = "https://github.com/data-literacy-alliance/dalia-curation/raw/refs/heads/main/curation/NFDI4Chem.csv"
    client = TeSSClient
    tess_oers = []
    for dalia_oer in read_dif13(url):
        if tess_oer := _from_dalia_dif13(dalia_oer):
            tess_oers.append(tess_oer)

            client.post(tess_oer)

    with open("/Users/cthoyt/Desktop/tess_from_dalia.json", "w") as file:
        json.dump(
            [tess_oer.model_dump(exclude_none=True, exclude_unset=True) for tess_oer in tess_oers],
            file,
            indent=2,
            ensure_ascii=False,
        )


def _main() -> None:
    payload = PostLearningMaterial(
        title="Test title",
        url="https://example.org/test",
        description="Test description",
        authors=["Charles Tapley Hoyt"],
    )

    base_url = "https://test.tesshub.hzdr.de"
    key = pystow.get_config("panosc", "test_key", raise_on_missing=True)
    email = pystow.get_config("panosc", "test_email")
    api_token = pystow.get_config("panosc", "test_api_token")
    client = TeSSClient(key=key, base_url=base_url)
    res = client.post(payload, api_key=api_token, email=email)
    res.raise_for_status()
    click.echo(json.dumps(res.json(), indent=2))


def _from_dalia_dif13(oer: EducationalResourceDIF13) -> LearningMaterial | None:
    if not oer.description:
        return None

    return LearningMaterial(
        slug=str(oer.uuid),  # the slug is the
        title=oer.title,
        url=oer.links[0],
        description=oer.description,
        keywords=oer.keywords,
        # resource_type, # TODO needs the DALIA-TeSS mapping
        # other_types
        scientific_topics=[
            Topic(
                label=get_discipline_label(discipline),
                uri=str(discipline),
            )
            for discipline in oer.disciplines
        ],
    )


if __name__ == "__main__":
    main()
