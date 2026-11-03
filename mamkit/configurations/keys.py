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

# Generated from the experiment configurations of earlier releases; see experiments/.

MACRO_F1_2_CLASSES = NAMESPACE.key("metric", "f1", "macro", "2-classes")
MACRO_F1_6_CLASSES = NAMESPACE.key("metric", "f1", "macro", "6-classes")
MARG_ARC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "marg", "arc", "audio-only", "bilstm-mfcc", "anonymous", "audio"
)
MARG_ARC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "audio-only", "bilstm-mfcc", "anonymous"
)
MARG_ARC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "marg", "arc", "audio-only", "bilstm-mfcc", "anonymous"
)
MARG_ARC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "marg", "arc", "audio-only", "bilstm-transformer", "anonymous", "audio"
)
MARG_ARC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "audio-only", "bilstm-transformer", "anonymous"
)
MARG_ARC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "marg", "arc", "audio-only", "bilstm-transformer", "anonymous"
)
MARG_ARC_AUDIO_ONLY_LOADER = NAMESPACE.key("loader", "marg", "arc", "audio-only")
MARG_ARC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "marg", "arc", "audio-only", "transformer-encoder", "anonymous", "audio"
)
MARG_ARC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "audio-only", "transformer-encoder", "anonymous"
)
MARG_ARC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "marg", "arc", "audio-only", "transformer-encoder", "anonymous"
)
MARG_ARC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "marg", "arc", "text-audio", "bilstm-mfcc", "anonymous", "audio"
)
MARG_ARC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT = NAMESPACE.key(
    "encoder", "marg", "arc", "text-audio", "bilstm-mfcc", "anonymous", "concat"
)
MARG_ARC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-audio", "bilstm-mfcc", "anonymous"
)
MARG_ARC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "marg", "arc", "text-audio", "bilstm-mfcc", "anonymous"
)
MARG_ARC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "marg", "arc", "text-audio", "bilstm-mfcc", "anonymous", "text"
)
MARG_ARC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "marg", "arc", "text-audio", "bilstm-transformer", "anonymous", "audio"
)
MARG_ARC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_CONCAT = NAMESPACE.key(
    "encoder", "marg", "arc", "text-audio", "bilstm-transformer", "anonymous", "concat"
)
MARG_ARC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-audio", "bilstm-transformer", "anonymous"
)
MARG_ARC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "marg", "arc", "text-audio", "bilstm-transformer", "anonymous"
)
MARG_ARC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "marg", "arc", "text-audio", "bilstm-transformer", "anonymous", "text"
)
MARG_ARC_TEXT_AUDIO_CSA_ANONYMOUS_CSA = NAMESPACE.key(
    "encoder", "marg", "arc", "text-audio", "csa", "anonymous", "csa"
)
MARG_ARC_TEXT_AUDIO_CSA_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-audio", "csa", "anonymous"
)
MARG_ARC_TEXT_AUDIO_CSA_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "marg", "arc", "text-audio", "csa", "anonymous"
)
MARG_ARC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "marg", "arc", "text-audio", "ensemble", "anonymous", "audio"
)
MARG_ARC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-audio", "ensemble", "anonymous", "audio-model"
)
MARG_ARC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-audio", "ensemble", "anonymous"
)
MARG_ARC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "marg", "arc", "text-audio", "ensemble", "anonymous"
)
MARG_ARC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "marg", "arc", "text-audio", "ensemble", "anonymous", "text"
)
MARG_ARC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-audio", "ensemble", "anonymous", "text-model"
)
MARG_ARC_TEXT_AUDIO_LOADER = NAMESPACE.key("loader", "marg", "arc", "text-audio")
MARG_ARC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_AUDIO = NAMESPACE.key(
    "encoder",
    "marg",
    "arc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "audio",
)
MARG_ARC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_CONCAT = NAMESPACE.key(
    "encoder",
    "marg",
    "arc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "concat",
)
MARG_ARC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-audio", "mmtransformer", "anonymous", "bert"
)
MARG_ARC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TASK = NAMESPACE.key(
    "task", "marg", "arc", "text-audio", "mmtransformer", "anonymous", "bert"
)
MARG_ARC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TEXT = NAMESPACE.key(
    "encoder", "marg", "arc", "text-audio", "mmtransformer", "anonymous", "bert", "text"
)
MARG_ARC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO = NAMESPACE.key(
    "encoder",
    "marg",
    "arc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "audio",
)
MARG_ARC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT = NAMESPACE.key(
    "encoder",
    "marg",
    "arc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "concat",
)
MARG_ARC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-audio", "mmtransformer", "anonymous", "roberta"
)
MARG_ARC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task", "marg", "arc", "text-audio", "mmtransformer", "anonymous", "roberta"
)
MARG_ARC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder",
    "marg",
    "arc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "text",
)
MARG_ARC_TEXT_AUDIO_MULTA_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-audio", "multa", "anonymous"
)
MARG_ARC_TEXT_AUDIO_MULTA_ANONYMOUS_MULTA = NAMESPACE.key(
    "encoder", "marg", "arc", "text-audio", "multa", "anonymous", "multa"
)
MARG_ARC_TEXT_AUDIO_MULTA_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "marg", "arc", "text-audio", "multa", "anonymous"
)
MARG_ARC_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-only", "bilstm", "anonymous"
)
MARG_ARC_TEXT_ONLY_BILSTM_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "marg", "arc", "text-only", "bilstm", "anonymous"
)
MARG_ARC_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "marg", "arc", "text-only", "bilstm", "anonymous", "text"
)
MARG_ARC_TEXT_ONLY_LOADER = NAMESPACE.key("loader", "marg", "arc", "text-only")
MARG_ARC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-only", "transformer", "anonymous", "bert"
)
MARG_ARC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TASK = NAMESPACE.key(
    "task", "marg", "arc", "text-only", "transformer", "anonymous", "bert"
)
MARG_ARC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TEXT = NAMESPACE.key(
    "encoder", "marg", "arc", "text-only", "transformer", "anonymous", "bert", "text"
)
MARG_ARC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model", "marg", "arc", "text-only", "transformer", "anonymous", "roberta"
)
MARG_ARC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task", "marg", "arc", "text-only", "transformer", "anonymous", "roberta"
)
MARG_ARC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder", "marg", "arc", "text-only", "transformer", "anonymous", "roberta", "text"
)
MARG_ENCODED_AUDIO_1 = NAMESPACE.key("featurizer", "marg", "encoded-audio", "1")
MARG_ENCODED_TEXT_1 = NAMESPACE.key("featurizer", "marg", "encoded-text", "1")
MARG_F1 = NAMESPACE.key("metric", "f1", "support-attack")
MARG_MFCC_1 = NAMESPACE.key("featurizer", "marg", "mfcc", "1")
MARG_TOKENIZED_1 = NAMESPACE.key("featurizer", "marg", "tokenized", "1")
MARG_TOKENIZED_2 = NAMESPACE.key("featurizer", "marg", "tokenized", "2")
MARG_VOCABULARY_1 = NAMESPACE.key("featurizer", "marg", "vocabulary", "1")
MMUSED_ACC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "mmused", "acc", "audio-only", "bilstm-mfcc", "anonymous", "audio"
)
MMUSED_ACC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "audio-only", "bilstm-mfcc", "anonymous"
)
MMUSED_ACC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "audio-only", "bilstm-mfcc", "anonymous"
)
MMUSED_ACC_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused",
    "acc",
    "audio-only",
    "bilstm-mfcc",
    "mancini-et-al-2022",
    "audio",
)
MMUSED_ACC_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "audio-only", "bilstm-mfcc", "mancini-et-al-2022"
)
MMUSED_ACC_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "audio-only", "bilstm-mfcc", "mancini-et-al-2022"
)
MMUSED_ACC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "mmused", "acc", "audio-only", "bilstm-transformer", "anonymous", "audio"
)
MMUSED_ACC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "audio-only", "bilstm-transformer", "anonymous"
)
MMUSED_ACC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "audio-only", "bilstm-transformer", "anonymous"
)
MMUSED_ACC_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused",
    "acc",
    "audio-only",
    "bilstm-transformer",
    "mancini-et-al-2022",
    "audio",
)
MMUSED_ACC_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "audio-only", "bilstm-transformer", "mancini-et-al-2022"
)
MMUSED_ACC_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "audio-only", "bilstm-transformer", "mancini-et-al-2022"
)
MMUSED_ACC_AUDIO_ONLY_LOADER = NAMESPACE.key("loader", "mmused", "acc", "audio-only")
MMUSED_ACC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused",
    "acc",
    "audio-only",
    "transformer-encoder",
    "anonymous",
    "audio",
)
MMUSED_ACC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "audio-only", "transformer-encoder", "anonymous"
)
MMUSED_ACC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "audio-only", "transformer-encoder", "anonymous"
)
MMUSED_ACC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-audio", "bilstm-mfcc", "anonymous", "audio"
)
MMUSED_ACC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-audio", "bilstm-mfcc", "anonymous", "concat"
)
MMUSED_ACC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-audio", "bilstm-mfcc", "anonymous"
)
MMUSED_ACC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "text-audio", "bilstm-mfcc", "anonymous"
)
MMUSED_ACC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-audio", "bilstm-mfcc", "anonymous", "text"
)
MMUSED_ACC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-audio", "bilstm-transformer", "anonymous", "audio"
)
MMUSED_ACC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused",
    "acc",
    "text-audio",
    "bilstm-transformer",
    "anonymous",
    "concat",
)
MMUSED_ACC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-audio", "bilstm-transformer", "anonymous"
)
MMUSED_ACC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "text-audio", "bilstm-transformer", "anonymous"
)
MMUSED_ACC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-audio", "bilstm-transformer", "anonymous", "text"
)
MMUSED_ACC_TEXT_AUDIO_CSA_ANONYMOUS_CSA = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-audio", "csa", "anonymous", "csa"
)
MMUSED_ACC_TEXT_AUDIO_CSA_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-audio", "csa", "anonymous"
)
MMUSED_ACC_TEXT_AUDIO_CSA_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "text-audio", "csa", "anonymous"
)
MMUSED_ACC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-audio", "ensemble", "anonymous", "audio"
)
MMUSED_ACC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-audio", "ensemble", "anonymous", "audio-model"
)
MMUSED_ACC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-audio", "ensemble", "anonymous"
)
MMUSED_ACC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "text-audio", "ensemble", "anonymous"
)
MMUSED_ACC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-audio", "ensemble", "anonymous", "text"
)
MMUSED_ACC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-audio", "ensemble", "anonymous", "text-model"
)
MMUSED_ACC_TEXT_AUDIO_LOADER = NAMESPACE.key("loader", "mmused", "acc", "text-audio")
MMUSED_ACC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused",
    "acc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "audio",
)
MMUSED_ACC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused",
    "acc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "concat",
)
MMUSED_ACC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-audio", "mmtransformer", "anonymous", "bert"
)
MMUSED_ACC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "text-audio", "mmtransformer", "anonymous", "bert"
)
MMUSED_ACC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TEXT = NAMESPACE.key(
    "encoder",
    "mmused",
    "acc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "text",
)
MMUSED_ACC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused",
    "acc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "audio",
)
MMUSED_ACC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused",
    "acc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "concat",
)
MMUSED_ACC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-audio", "mmtransformer", "anonymous", "roberta"
)
MMUSED_ACC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "text-audio", "mmtransformer", "anonymous", "roberta"
)
MMUSED_ACC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder",
    "mmused",
    "acc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "text",
)
MMUSED_ACC_TEXT_AUDIO_MULTA_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-audio", "multa", "anonymous"
)
MMUSED_ACC_TEXT_AUDIO_MULTA_ANONYMOUS_MULTA = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-audio", "multa", "anonymous", "multa"
)
MMUSED_ACC_TEXT_AUDIO_MULTA_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "text-audio", "multa", "anonymous"
)
MMUSED_ACC_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-only", "bilstm", "anonymous"
)
MMUSED_ACC_TEXT_ONLY_BILSTM_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "text-only", "bilstm", "anonymous"
)
MMUSED_ACC_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-only", "bilstm", "anonymous", "text"
)
MMUSED_ACC_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-only", "bilstm", "mancini-et-al-2022"
)
MMUSED_ACC_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "text-only", "bilstm", "mancini-et-al-2022"
)
MMUSED_ACC_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_TEXT = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-only", "bilstm", "mancini-et-al-2022", "text"
)
MMUSED_ACC_TEXT_ONLY_LOADER = NAMESPACE.key("loader", "mmused", "acc", "text-only")
MMUSED_ACC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-only", "transformer", "anonymous", "bert"
)
MMUSED_ACC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "text-only", "transformer", "anonymous", "bert"
)
MMUSED_ACC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TEXT = NAMESPACE.key(
    "encoder", "mmused", "acc", "text-only", "transformer", "anonymous", "bert", "text"
)
MMUSED_ACC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model", "mmused", "acc", "text-only", "transformer", "anonymous", "roberta"
)
MMUSED_ACC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task", "mmused", "acc", "text-only", "transformer", "anonymous", "roberta"
)
MMUSED_ACC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder",
    "mmused",
    "acc",
    "text-only",
    "transformer",
    "anonymous",
    "roberta",
    "text",
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "mmused", "asd", "audio-only", "bilstm-mfcc", "anonymous", "audio"
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "audio-only", "bilstm-mfcc", "anonymous"
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "audio-only", "bilstm-mfcc", "anonymous"
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused",
    "asd",
    "audio-only",
    "bilstm-mfcc",
    "mancini-et-al-2022",
    "audio",
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "audio-only", "bilstm-mfcc", "mancini-et-al-2022"
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "audio-only", "bilstm-mfcc", "mancini-et-al-2022"
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "mmused", "asd", "audio-only", "bilstm-transformer", "anonymous", "audio"
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "audio-only", "bilstm-transformer", "anonymous"
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "audio-only", "bilstm-transformer", "anonymous"
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused",
    "asd",
    "audio-only",
    "bilstm-transformer",
    "mancini-et-al-2022",
    "audio",
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "audio-only", "bilstm-transformer", "mancini-et-al-2022"
)
MMUSED_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "audio-only", "bilstm-transformer", "mancini-et-al-2022"
)
MMUSED_ASD_AUDIO_ONLY_LOADER = NAMESPACE.key("loader", "mmused", "asd", "audio-only")
MMUSED_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused",
    "asd",
    "audio-only",
    "transformer-encoder",
    "anonymous",
    "audio",
)
MMUSED_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "audio-only", "transformer-encoder", "anonymous"
)
MMUSED_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "audio-only", "transformer-encoder", "anonymous"
)
MMUSED_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-audio", "bilstm-mfcc", "anonymous", "audio"
)
MMUSED_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-audio", "bilstm-mfcc", "anonymous", "concat"
)
MMUSED_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-audio", "bilstm-mfcc", "anonymous"
)
MMUSED_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "text-audio", "bilstm-mfcc", "anonymous"
)
MMUSED_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-audio", "bilstm-mfcc", "anonymous", "text"
)
MMUSED_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-audio", "bilstm-transformer", "anonymous", "audio"
)
MMUSED_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused",
    "asd",
    "text-audio",
    "bilstm-transformer",
    "anonymous",
    "concat",
)
MMUSED_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-audio", "bilstm-transformer", "anonymous"
)
MMUSED_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "text-audio", "bilstm-transformer", "anonymous"
)
MMUSED_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-audio", "bilstm-transformer", "anonymous", "text"
)
MMUSED_ASD_TEXT_AUDIO_CSA_ANONYMOUS_CSA = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-audio", "csa", "anonymous", "csa"
)
MMUSED_ASD_TEXT_AUDIO_CSA_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-audio", "csa", "anonymous"
)
MMUSED_ASD_TEXT_AUDIO_CSA_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "text-audio", "csa", "anonymous"
)
MMUSED_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-audio", "ensemble", "anonymous", "audio"
)
MMUSED_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-audio", "ensemble", "anonymous", "audio-model"
)
MMUSED_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-audio", "ensemble", "anonymous"
)
MMUSED_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "text-audio", "ensemble", "anonymous"
)
MMUSED_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-audio", "ensemble", "anonymous", "text"
)
MMUSED_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-audio", "ensemble", "anonymous", "text-model"
)
MMUSED_ASD_TEXT_AUDIO_LOADER = NAMESPACE.key("loader", "mmused", "asd", "text-audio")
MMUSED_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "audio",
)
MMUSED_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "concat",
)
MMUSED_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-audio", "mmtransformer", "anonymous", "bert"
)
MMUSED_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "text-audio", "mmtransformer", "anonymous", "bert"
)
MMUSED_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TEXT = NAMESPACE.key(
    "encoder",
    "mmused",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "text",
)
MMUSED_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "audio",
)
MMUSED_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "concat",
)
MMUSED_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-audio", "mmtransformer", "anonymous", "roberta"
)
MMUSED_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "text-audio", "mmtransformer", "anonymous", "roberta"
)
MMUSED_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder",
    "mmused",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "text",
)
MMUSED_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-audio", "multa", "anonymous"
)
MMUSED_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_MULTA = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-audio", "multa", "anonymous", "multa"
)
MMUSED_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "text-audio", "multa", "anonymous"
)
MMUSED_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-only", "bilstm", "anonymous"
)
MMUSED_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "text-only", "bilstm", "anonymous"
)
MMUSED_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-only", "bilstm", "anonymous", "text"
)
MMUSED_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-only", "bilstm", "mancini-et-al-2022"
)
MMUSED_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "text-only", "bilstm", "mancini-et-al-2022"
)
MMUSED_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_TEXT = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-only", "bilstm", "mancini-et-al-2022", "text"
)
MMUSED_ASD_TEXT_ONLY_LOADER = NAMESPACE.key("loader", "mmused", "asd", "text-only")
MMUSED_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-only", "transformer", "anonymous", "bert"
)
MMUSED_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "text-only", "transformer", "anonymous", "bert"
)
MMUSED_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TEXT = NAMESPACE.key(
    "encoder", "mmused", "asd", "text-only", "transformer", "anonymous", "bert", "text"
)
MMUSED_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model", "mmused", "asd", "text-only", "transformer", "anonymous", "roberta"
)
MMUSED_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task", "mmused", "asd", "text-only", "transformer", "anonymous", "roberta"
)
MMUSED_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder",
    "mmused",
    "asd",
    "text-only",
    "transformer",
    "anonymous",
    "roberta",
    "text",
)
MMUSED_ENCODED_AUDIO_1 = NAMESPACE.key("featurizer", "mmused", "encoded-audio", "1")
MMUSED_ENCODED_AUDIO_2 = NAMESPACE.key("featurizer", "mmused", "encoded-audio", "2")
MMUSED_ENCODED_TEXT_1 = NAMESPACE.key("featurizer", "mmused", "encoded-text", "1")
MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "audio-only",
    "bilstm-mfcc",
    "anonymous",
    "audio",
)
MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "audio-only", "bilstm-mfcc", "anonymous"
)
MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "audio-only", "bilstm-mfcc", "anonymous"
)
MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "audio-only",
    "bilstm-transformer",
    "anonymous",
    "audio",
)
MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "audio-only", "bilstm-transformer", "anonymous"
)
MMUSED_FALLACY_AFC_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "audio-only", "bilstm-transformer", "anonymous"
)
MMUSED_FALLACY_AFC_AUDIO_ONLY_LOADER = NAMESPACE.key(
    "loader", "mmused-fallacy", "afc", "audio-only"
)
MMUSED_FALLACY_AFC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "audio-only",
    "transformer-encoder",
    "anonymous",
    "audio",
)
MMUSED_FALLACY_AFC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "audio-only", "transformer-encoder", "anonymous"
)
MMUSED_FALLACY_AFC_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "audio-only", "transformer-encoder", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "bilstm-mfcc",
    "anonymous",
    "audio",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "bilstm-mfcc",
    "anonymous",
    "concat",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "text-audio", "bilstm-mfcc", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "text-audio", "bilstm-mfcc", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused-fallacy", "afc", "text-audio", "bilstm-mfcc", "anonymous", "text"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "bilstm-transformer",
    "anonymous",
    "audio",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "bilstm-transformer",
    "anonymous",
    "concat",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "text-audio", "bilstm-transformer", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "text-audio", "bilstm-transformer", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "bilstm-transformer",
    "anonymous",
    "text",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_CSA_ANONYMOUS_CSA = NAMESPACE.key(
    "encoder", "mmused-fallacy", "afc", "text-audio", "csa", "anonymous", "csa"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_CSA_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "text-audio", "csa", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_CSA_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "text-audio", "csa", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "mmused-fallacy", "afc", "text-audio", "ensemble", "anonymous", "audio"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO_MODEL = NAMESPACE.key(
    "model",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "ensemble",
    "anonymous",
    "audio-model",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "text-audio", "ensemble", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "text-audio", "ensemble", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused-fallacy", "afc", "text-audio", "ensemble", "anonymous", "text"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT_MODEL = NAMESPACE.key(
    "model",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "ensemble",
    "anonymous",
    "text-model",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_LOADER = NAMESPACE.key(
    "loader", "mmused-fallacy", "afc", "text-audio"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "audio",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "concat",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "text-audio", "mmtransformer", "anonymous", "bert"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "text-audio", "mmtransformer", "anonymous", "bert"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TEXT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "text",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "audio",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "concat",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "text",
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MULTA_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "text-audio", "multa", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MULTA_ANONYMOUS_MULTA = NAMESPACE.key(
    "encoder", "mmused-fallacy", "afc", "text-audio", "multa", "anonymous", "multa"
)
MMUSED_FALLACY_AFC_TEXT_AUDIO_MULTA_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "text-audio", "multa", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "text-only", "bilstm", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_ONLY_BILSTM_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "text-only", "bilstm", "anonymous"
)
MMUSED_FALLACY_AFC_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused-fallacy", "afc", "text-only", "bilstm", "anonymous", "text"
)
MMUSED_FALLACY_AFC_TEXT_ONLY_LOADER = NAMESPACE.key(
    "loader", "mmused-fallacy", "afc", "text-only"
)
MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "text-only", "transformer", "anonymous", "bert"
)
MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "text-only", "transformer", "anonymous", "bert"
)
MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TEXT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-only",
    "transformer",
    "anonymous",
    "bert",
    "text",
)
MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afc", "text-only", "transformer", "anonymous", "roberta"
)
MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afc", "text-only", "transformer", "anonymous", "roberta"
)
MMUSED_FALLACY_AFC_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afc",
    "text-only",
    "transformer",
    "anonymous",
    "roberta",
    "text",
)
MMUSED_FALLACY_AFD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afd",
    "audio-only",
    "bilstm-mfcc",
    "anonymous",
    "audio",
)
MMUSED_FALLACY_AFD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afd", "audio-only", "bilstm-mfcc", "anonymous"
)
MMUSED_FALLACY_AFD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afd", "audio-only", "bilstm-mfcc", "anonymous"
)
MMUSED_FALLACY_AFD_AUDIO_ONLY_LOADER = NAMESPACE.key(
    "loader", "mmused-fallacy", "afd", "audio-only"
)
MMUSED_FALLACY_AFD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afd",
    "audio-only",
    "transformer-encoder",
    "anonymous",
    "audio",
)
MMUSED_FALLACY_AFD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afd", "audio-only", "transformer-encoder", "anonymous"
)
MMUSED_FALLACY_AFD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afd", "audio-only", "transformer-encoder", "anonymous"
)
MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afd",
    "text-audio",
    "bilstm-mfcc",
    "anonymous",
    "audio",
)
MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afd",
    "text-audio",
    "bilstm-mfcc",
    "anonymous",
    "concat",
)
MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afd", "text-audio", "bilstm-mfcc", "anonymous"
)
MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afd", "text-audio", "bilstm-mfcc", "anonymous"
)
MMUSED_FALLACY_AFD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused-fallacy", "afd", "text-audio", "bilstm-mfcc", "anonymous", "text"
)
MMUSED_FALLACY_AFD_TEXT_AUDIO_LOADER = NAMESPACE.key(
    "loader", "mmused-fallacy", "afd", "text-audio"
)
MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "audio",
)
MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "concat",
)
MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model",
    "mmused-fallacy",
    "afd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
)
MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task",
    "mmused-fallacy",
    "afd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
)
MMUSED_FALLACY_AFD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "text",
)
MMUSED_FALLACY_AFD_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afd", "text-only", "bilstm", "anonymous"
)
MMUSED_FALLACY_AFD_TEXT_ONLY_BILSTM_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afd", "text-only", "bilstm", "anonymous"
)
MMUSED_FALLACY_AFD_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "mmused-fallacy", "afd", "text-only", "bilstm", "anonymous", "text"
)
MMUSED_FALLACY_AFD_TEXT_ONLY_LOADER = NAMESPACE.key(
    "loader", "mmused-fallacy", "afd", "text-only"
)
MMUSED_FALLACY_AFD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model", "mmused-fallacy", "afd", "text-only", "transformer", "anonymous", "roberta"
)
MMUSED_FALLACY_AFD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task", "mmused-fallacy", "afd", "text-only", "transformer", "anonymous", "roberta"
)
MMUSED_FALLACY_AFD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder",
    "mmused-fallacy",
    "afd",
    "text-only",
    "transformer",
    "anonymous",
    "roberta",
    "text",
)
MMUSED_FALLACY_ENCODED_AUDIO_1 = NAMESPACE.key(
    "featurizer", "mmused-fallacy", "encoded-audio", "1"
)
MMUSED_FALLACY_ENCODED_AUDIO_2 = NAMESPACE.key(
    "featurizer", "mmused-fallacy", "encoded-audio", "2"
)
MMUSED_FALLACY_ENCODED_TEXT_1 = NAMESPACE.key(
    "featurizer", "mmused-fallacy", "encoded-text", "1"
)
MMUSED_FALLACY_MFCC_1 = NAMESPACE.key("featurizer", "mmused-fallacy", "mfcc", "1")
MMUSED_FALLACY_TOKENIZED_1 = NAMESPACE.key(
    "featurizer", "mmused-fallacy", "tokenized", "1"
)
MMUSED_FALLACY_TOKENIZED_2 = NAMESPACE.key(
    "featurizer", "mmused-fallacy", "tokenized", "2"
)
MMUSED_FALLACY_VOCABULARY_1 = NAMESPACE.key(
    "featurizer", "mmused-fallacy", "vocabulary", "1"
)
MMUSED_MFCC_1 = NAMESPACE.key("featurizer", "mmused", "mfcc", "1")
MMUSED_MFCC_2 = NAMESPACE.key("featurizer", "mmused", "mfcc", "2")
MMUSED_TOKENIZED_1 = NAMESPACE.key("featurizer", "mmused", "tokenized", "1")
MMUSED_TOKENIZED_2 = NAMESPACE.key("featurizer", "mmused", "tokenized", "2")
MMUSED_VOCABULARY_1 = NAMESPACE.key("featurizer", "mmused", "vocabulary", "1")
MMUSED_VOCABULARY_2 = NAMESPACE.key("featurizer", "mmused", "vocabulary", "2")
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "ukdebates", "asd", "audio-only", "bilstm-mfcc", "anonymous", "audio"
)
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "audio-only", "bilstm-mfcc", "anonymous"
)
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "audio-only", "bilstm-mfcc", "anonymous"
)
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_AUDIO = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "audio-only",
    "bilstm-mfcc",
    "mancini-et-al-2022",
    "audio",
)
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "audio-only", "bilstm-mfcc", "mancini-et-al-2022"
)
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_MFCC_MANCINI_ET_AL_2022_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "audio-only", "bilstm-mfcc", "mancini-et-al-2022"
)
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "audio-only",
    "bilstm-transformer",
    "anonymous",
    "audio",
)
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "audio-only", "bilstm-transformer", "anonymous"
)
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "audio-only", "bilstm-transformer", "anonymous"
)
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_AUDIO = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "audio-only",
    "bilstm-transformer",
    "mancini-et-al-2022",
    "audio",
)
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_MODEL = NAMESPACE.key(
    "model",
    "ukdebates",
    "asd",
    "audio-only",
    "bilstm-transformer",
    "mancini-et-al-2022",
)
UKDEBATES_ASD_AUDIO_ONLY_BILSTM_TRANSFORMER_MANCINI_ET_AL_2022_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "audio-only", "bilstm-transformer", "mancini-et-al-2022"
)
UKDEBATES_ASD_AUDIO_ONLY_LOADER = NAMESPACE.key(
    "loader", "ukdebates", "asd", "audio-only"
)
UKDEBATES_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "audio-only",
    "transformer-encoder",
    "anonymous",
    "audio",
)
UKDEBATES_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "audio-only", "transformer-encoder", "anonymous"
)
UKDEBATES_ASD_AUDIO_ONLY_TRANSFORMER_ENCODER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "audio-only", "transformer-encoder", "anonymous"
)
UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "ukdebates", "asd", "text-audio", "bilstm-mfcc", "anonymous", "audio"
)
UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_CONCAT = NAMESPACE.key(
    "encoder", "ukdebates", "asd", "text-audio", "bilstm-mfcc", "anonymous", "concat"
)
UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-audio", "bilstm-mfcc", "anonymous"
)
UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "text-audio", "bilstm-mfcc", "anonymous"
)
UKDEBATES_ASD_TEXT_AUDIO_BILSTM_MFCC_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "ukdebates", "asd", "text-audio", "bilstm-mfcc", "anonymous", "text"
)
UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "text-audio",
    "bilstm-transformer",
    "anonymous",
    "audio",
)
UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_CONCAT = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "text-audio",
    "bilstm-transformer",
    "anonymous",
    "concat",
)
UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-audio", "bilstm-transformer", "anonymous"
)
UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "text-audio", "bilstm-transformer", "anonymous"
)
UKDEBATES_ASD_TEXT_AUDIO_BILSTM_TRANSFORMER_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "text-audio",
    "bilstm-transformer",
    "anonymous",
    "text",
)
UKDEBATES_ASD_TEXT_AUDIO_CSA_ANONYMOUS_CSA = NAMESPACE.key(
    "encoder", "ukdebates", "asd", "text-audio", "csa", "anonymous", "csa"
)
UKDEBATES_ASD_TEXT_AUDIO_CSA_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-audio", "csa", "anonymous"
)
UKDEBATES_ASD_TEXT_AUDIO_CSA_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "text-audio", "csa", "anonymous"
)
UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO = NAMESPACE.key(
    "encoder", "ukdebates", "asd", "text-audio", "ensemble", "anonymous", "audio"
)
UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_AUDIO_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-audio", "ensemble", "anonymous", "audio-model"
)
UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-audio", "ensemble", "anonymous"
)
UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "text-audio", "ensemble", "anonymous"
)
UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "ukdebates", "asd", "text-audio", "ensemble", "anonymous", "text"
)
UKDEBATES_ASD_TEXT_AUDIO_ENSEMBLE_ANONYMOUS_TEXT_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-audio", "ensemble", "anonymous", "text-model"
)
UKDEBATES_ASD_TEXT_AUDIO_LOADER = NAMESPACE.key(
    "loader", "ukdebates", "asd", "text-audio"
)
UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_AUDIO = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "audio",
)
UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_CONCAT = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "concat",
)
UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-audio", "mmtransformer", "anonymous", "bert"
)
UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "text-audio", "mmtransformer", "anonymous", "bert"
)
UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_BERT_TEXT = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "bert",
    "text",
)
UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_AUDIO = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "audio",
)
UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_CONCAT = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "concat",
)
UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-audio", "mmtransformer", "anonymous", "roberta"
)
UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "text-audio", "mmtransformer", "anonymous", "roberta"
)
UKDEBATES_ASD_TEXT_AUDIO_MMTRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "text-audio",
    "mmtransformer",
    "anonymous",
    "roberta",
    "text",
)
UKDEBATES_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-audio", "multa", "anonymous"
)
UKDEBATES_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_MULTA = NAMESPACE.key(
    "encoder", "ukdebates", "asd", "text-audio", "multa", "anonymous", "multa"
)
UKDEBATES_ASD_TEXT_AUDIO_MULTA_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "text-audio", "multa", "anonymous"
)
UKDEBATES_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-only", "bilstm", "anonymous"
)
UKDEBATES_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "text-only", "bilstm", "anonymous"
)
UKDEBATES_ASD_TEXT_ONLY_BILSTM_ANONYMOUS_TEXT = NAMESPACE.key(
    "encoder", "ukdebates", "asd", "text-only", "bilstm", "anonymous", "text"
)
UKDEBATES_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-only", "bilstm", "mancini-et-al-2022"
)
UKDEBATES_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "text-only", "bilstm", "mancini-et-al-2022"
)
UKDEBATES_ASD_TEXT_ONLY_BILSTM_MANCINI_ET_AL_2022_TEXT = NAMESPACE.key(
    "encoder", "ukdebates", "asd", "text-only", "bilstm", "mancini-et-al-2022", "text"
)
UKDEBATES_ASD_TEXT_ONLY_LOADER = NAMESPACE.key(
    "loader", "ukdebates", "asd", "text-only"
)
UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-only", "transformer", "anonymous", "bert"
)
UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "text-only", "transformer", "anonymous", "bert"
)
UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_BERT_TEXT = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "text-only",
    "transformer",
    "anonymous",
    "bert",
    "text",
)
UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_MODEL = NAMESPACE.key(
    "model", "ukdebates", "asd", "text-only", "transformer", "anonymous", "roberta"
)
UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TASK = NAMESPACE.key(
    "task", "ukdebates", "asd", "text-only", "transformer", "anonymous", "roberta"
)
UKDEBATES_ASD_TEXT_ONLY_TRANSFORMER_ANONYMOUS_ROBERTA_TEXT = NAMESPACE.key(
    "encoder",
    "ukdebates",
    "asd",
    "text-only",
    "transformer",
    "anonymous",
    "roberta",
    "text",
)
UKDEBATES_ENCODED_AUDIO_1 = NAMESPACE.key(
    "featurizer", "ukdebates", "encoded-audio", "1"
)
UKDEBATES_ENCODED_AUDIO_2 = NAMESPACE.key(
    "featurizer", "ukdebates", "encoded-audio", "2"
)
UKDEBATES_ENCODED_AUDIO_3 = NAMESPACE.key(
    "featurizer", "ukdebates", "encoded-audio", "3"
)
UKDEBATES_ENCODED_TEXT_1 = NAMESPACE.key("featurizer", "ukdebates", "encoded-text", "1")
UKDEBATES_F1 = NAMESPACE.key("metric", "f1", "binary")
UKDEBATES_MFCC_1 = NAMESPACE.key("featurizer", "ukdebates", "mfcc", "1")
UKDEBATES_MFCC_2 = NAMESPACE.key("featurizer", "ukdebates", "mfcc", "2")
UKDEBATES_TOKENIZED_1 = NAMESPACE.key("featurizer", "ukdebates", "tokenized", "1")
UKDEBATES_TOKENIZED_2 = NAMESPACE.key("featurizer", "ukdebates", "tokenized", "2")
UKDEBATES_VOCABULARY_1 = NAMESPACE.key("featurizer", "ukdebates", "vocabulary", "1")
