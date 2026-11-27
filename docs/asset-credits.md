# Assets and model provenance

- Barlow Condensed, IBM Plex Sans and IBM Plex Mono: Google Fonts, SIL Open Font License. License files are bundled beside the self-hosted font assets.
- Sample portrait: NASA photograph of Eileen Collins, public domain, distributed by scikit-image. Source: https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/astronaut.png . The scikit-image data documentation identifies the source and public-domain status: https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/data/_fetchers.py . No endorsement is implied.
- Brand mark and face-study line illustration: original project vector artwork.
- Face location: OpenCV’s bundled Haar Cascade, retained from the initial project.
- Expression classifier: project-specific CNN trained from random initialization on FER-2013. No third-party expression-classification weights are used. Training dataset source: https://www.kaggle.com/datasets/msambare/fer2013 . Dataset images are not distributed with this repository.

- Improved annotations: Microsoft FERPlus, pinned commit `ae2128abf776409c93e50cf7c9d87180673314e6`, https://github.com/microsoft/FERPlus . Original-order FER2013 image CSV mirror: https://www.kaggle.com/datasets/deadskull7/fer2013 . Crowdsourced votes are aligned by their documented row order and split labels.
