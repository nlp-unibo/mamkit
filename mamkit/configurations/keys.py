"""Every key MAMKit registers, declared once."""

from cinnamon.registry import Namespace

NAMESPACE = Namespace("mamkit")

ALL_TRAIN_SPLIT = NAMESPACE.key("split", "all-train")
MMUSED_SPLIT = NAMESPACE.key("split", "mmused")
UKDEBATES_MANCINI_2022_SPLIT = NAMESPACE.key("split", "ukdebates", "mancini-et-al-2022")
MARG_085_MANCINI_2022_SPLIT = NAMESPACE.key(
    "split", "marg", "confidence-0.85", "mancini-et-al-2022"
)
MMUSED_FALLACY_MANCINI_2024_SPLIT = NAMESPACE.key(
    "split", "mmused-fallacy", "mancini-et-al-2024"
)
MMUSED_FALLACY_MM_ARGFALLACY_2025_SPLIT = NAMESPACE.key(
    "split", "mmused-fallacy", "mm-argfallacy-2025"
)
