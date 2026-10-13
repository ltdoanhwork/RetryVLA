from __future__ import annotations

import logging
import warnings


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s | %(message)s", force=True)
    for noisy in ("httpx", "huggingface_hub", "urllib3", "robosuite", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
