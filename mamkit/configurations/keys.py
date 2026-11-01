"""Every key MAMKit registers, declared once."""

from cinnamon.registry import Namespace

NAMESPACE = Namespace("mamkit")

ALL_TRAIN_SPLIT = NAMESPACE.key("split", "all-train")
COLUMN_SPLIT = NAMESPACE.key("split", "column")
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

TOY_LOADER = NAMESPACE.key("loader", "toy")
VOCABULARY_TEXT = NAMESPACE.key("featurizer", "text", "vocabulary")
TOY_TEXT_BILSTM = NAMESPACE.key("encoder", "text-bilstm", "toy")
TOY_CLASSIFIER = NAMESPACE.key("model", "classifier", "toy")
TOY_F1 = NAMESPACE.key("metric", "f1", "toy")
TOY_TASK = NAMESPACE.key("task", "toy")
