"""Utilities and constants."""

from pathlib import Path

__all__ = ["OUTPUT_DIR"]

from rdflib import Namespace, URIRef

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent.parent.resolve()
OUTPUT_DIR = ROOT.joinpath("output")
OUTPUT_DIR.mkdir(exist_ok=True)


LICENSE_ONT = Namespace("https://w3id.org/license-ontology/")
UNSPECIFIED_OR_PROPRIETARY = LICENSE_ONT["unspecified"]
TESS_TO_LICENSE: dict[str, URIRef] = {
    "notspecified": UNSPECIFIED_OR_PROPRIETARY,
    "other-at": LICENSE_ONT["requires-attribution"],  # attribution
    "other-closed": LICENSE_ONT["not-open"],  # not open
    "other-open": LICENSE_ONT["open"],  # open
    "other-nc": LICENSE_ONT["non-commercial"],  # not commercial
    "other-pd": LICENSE_ONT["public-domain"],  # public domain
}
