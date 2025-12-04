from oerbservatory.sources.dalia import get_dalia
from oerbservatory.sources.gtn import get_gtn
from oerbservatory.sources.oerhub import get_oerhub
from oerbservatory.sources.oersi import get_oersi
import click
from tess_downloader import TeSSClient


@click.command()
def main():
    """Upload content to various mTeSS-X instances."""
    xx = [
        (get_dalia, "dalia"),
        (get_gtn, "kcd"),
        (get_oerhub, "oerhub"),
        (get_oersi, "oersi"),
    ]
    for func, key in xx:
        base_url = f"https://{key}.tesshub.hzdr.de/"
        client = TeSSClient(key=key, base_url=base_url)
        resources = func()
        for resource in resources:
            client.post(resource)
