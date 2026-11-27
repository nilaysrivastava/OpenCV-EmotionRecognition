import re
import urllib.request
from pathlib import Path

url = "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap"
css = (
    urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 Chrome/130.0.0.0 Safari/537.36"}
        )
    )
    .read()
    .decode()
)
for name, family, weight in [
    ("display-condensed", "Barlow Condensed", "600"),
    ("body-plex", "IBM Plex Sans", "400"),
    ("mono-plex", "IBM Plex Mono", "400"),
]:
    blocks = [
        b
        for b in css.split("}")
        if "font-family: '" + family + "'" in b and "font-weight: " + weight + ";" in b
    ]
    b = blocks[-1]
    u = re.search(r"url\(([^)]+)\)", b).group(1)
    Path("static/assets/" + name + ".ttf").write_bytes(urllib.request.urlopen(u).read())
for family in ["barlowcondensed", "ibmplexsans", "ibmplexmono"]:
    Path("static/assets/" + family + "-OFL.txt").write_bytes(
        urllib.request.urlopen(
            "https://raw.githubusercontent.com/google/fonts/main/ofl/"
            + family
            + "/OFL.txt"
        ).read()
    )
print("Self-hosted Barlow Condensed, IBM Plex Sans and IBM Plex Mono installed.")
