<p align="center">
    <br>
    <img src="figures/mamkit-logo-image.png" width="500"/>
    <br>
<p>

<div align="center">

| 🌐 [Website](https://nlp-unibo.github.io/mamkit/) | 📚 [Documentation](https://nlp-unibo.github.io/mamkit/mamkit.html) | 🤝 [Contributing](https://nlp-unibo.github.io/mamkit/contribute.html) | 
</div>

# MAMKit: Multimodal Argument Mining Toolkit
A Comprehensive Multimodal Argument Mining Toolkit. 



## Table of Contents
- [Introduction](#introduction)
- [Installation](#-installation)
- [Usage](#%EF%B8%8F-usage)
  - [Run a registered experiment](#run-a-registered-experiment)
  - [Run an experiment from Python](#run-an-experiment-from-python)
  - [Select experiments by tag](#select-experiments-by-tag)
  - [Add a dataset or a model](#add-a-dataset-or-a-model)
- [Structure](#-structure)
- [Documentation](#-website-and-documentation)
- [Contributing](#-contributing)
- [Contact Us](#-contact-us)
- [Citation](#-citation)

## Introduction
MAMKit is an open-source, publicly available PyTorch toolkit designed to access and develop 
datasets, models, and benchmarks for Multimodal Argument Mining (MAM).
It provides a flexible interface for accessing and integrating datasets, models, and preprocessing strategies through 
composition or custom definition. 
MAMKit is designed to be extendible, ensure replicability, and provide a shared interface as a common foundation for
experimentation in the field.

At the time of writing, MAMKit offers 4 datasets, 4 tasks and 6 distinct model architectures, along with audio and text 
processing capabilities, organized in 5 main components.

| **Datasets**                                                             | **Tasks**                                                                               |
|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [UkDebates](https://doi.org/10.1145/2850417)                             | Argumentative Sentence Detection (ASD)                                                  |
| [MArg<sub>&gamma;</sub>](https://doi.org/10.18653/v1/2021.argmining-1.8) | Argumentative Relation Classification (ARC)                                             |
| [MM-USED](https://aclanthology.org/2022.argmining-1.15)                  | Argumentative Sentence Detection (ASD) <br>Argumentative Component Classification (ACC) |
| [MM-USED-fallacy](https://aclanthology.org/2024.eacl-short.16)                                                      | Argumentative Fallacy Classification (AFC) <br> Argumentative Fallacy Detection(AFD)                                            |


| **Model**  | **Text Encoding** | **Audio Encoding**                          | **Fusion**   |
|------------|-------------------|---------------------------------------------|--------------|
| BiLSTM     | GloVe + BiLSTM    | (Wav2Vec2 &or; MFCCs) + BiLSTM                | Concat-Late  |
| MM-BERT    | BERT              | (Wav2Vec2 &or; HuBERT &or; WavLM) + BiLSTM      | Concat-Late  |
| MM-RoBERTa | RoBERTa           | (Wav2Vec2 &or; HuBERT &or; WavLM) + BiLSTM      | Concat-Late  |
| CSA        | BERT              | (Wav2Vec2 &or; HuBERT &or; WavLM) + Transformer | Concat-Early |
| Ensemble   | BERT              | (Wav2Vec2 &or; HuBERT &or; WavLM) + Transformer | Avg-Late     |
| Mul_TA     | BERT              | (Wav2Vec2 &or; HuBERT &or; WavLM) + Transformer | Cross        |


## 🔧 Installation


### ⚠️ Prerequisites
Before installing MAMKit, ensure you have the following:

- **Python 3.10 to 3.13** (MAMKit is tested with these versions)
- **ffmpeg** (Required for audio processing)  
  You can install it via:
  ```bash
  sudo apt install ffmpeg  # Debian/Ubuntu  
  brew install ffmpeg      # macOS  
  choco install ffmpeg     # Windows (using Chocolatey)  
  ```

For other installation methods, refer to the [FFmpeg official website](https://www.ffmpeg.org/). 


### Install via PyPi

1. Install MAMKit using PyPi:
```bash
pip install mamkit 

```
2. Access MAMKit in your Python code: 
```bash
import mamkit 
```

### Install from GitHub
This installation is recommended for users who wish to conduct experiments and customize the toolkit according to their needs.

1. Clone the repository and install it in editable mode:
```bash
git clone git@github.com:nlp-unibo/mamkit.git
cd mamkit
pip install -e .
```
2. Access MAMKit in your Python code: 
```bash
import mamkit 
```


## ⚙️ Usage
MAMKit is built on [Cinnamon](https://nlp.unibo.it/cinnamon/).
Each corpus, split strategy, featurizer, model, and metric is a component with a registered configuration.
Each experiment of the paper is a registered task that combines them.
A task trains a fresh model on every split of its corpus, once per seed, and reports the mean and standard deviation of every metric.

### Run a registered experiment
The `cmn-run` command lists the registered tasks, asks which ones to run, and runs them.
It needs the `cli` extra of Cinnamon:

```bash
pip install "cinnamon-core[cli]"
```

From a clone of the repository, point `cmn-run` to the package directory:

```bash
cmn-run -dir mamkit
```

From a PyPI installation, point it to the installed package:

```bash
cmn-run -dir "$(python -c 'import mamkit, os; print(os.path.dirname(mamkit.__file__))')"
```

The command asks for a namespace, a name, and tags.
Select the `task` name, then add tags until the list holds the experiments to run.
Each task writes `results.json` under its `save_path`, which is relative to the working directory.
For example, the BiLSTM of Mancini et al. (2022) on UKDebates writes to `results/ukdebates/asd/text-only/bilstm/mancini-et-al-2022`.
On first use, each corpus is downloaded into a folder of the working directory, such as `UKDebates/`.

### Run an experiment from Python
A task can also be built and run from a script:

```python
from pathlib import Path

import mamkit
from cinnamon.registry import RegistrationKey, Registry

Registry.build(directory=Path(mamkit.__file__).parent)
key = RegistrationKey(
    name="task",
    tags={"ukdebates", "asd", "text-only", "bilstm", "mancini-et-al-2022"},
    namespace="mamkit",
)
results = Registry.from_key(key).run()
print(results["test"])
```

`Registry.retrieve_configuration(key)` returns the configuration without building the task, so its fields can be inspected or changed first.

### Select experiments by tag
Every task key has the name `task` and the namespace `mamkit`.
Its tags name the corpus, the task, the input mode, the model family, and the source of the configuration:

| **Field**      | **Tags**                                                                                                         |
|----------------|------------------------------------------------------------------------------------------------------------------|
| Corpus         | `ukdebates`, `marg`, `mmused`, `mmused-fallacy`                                                                   |
| Task           | `asd`, `arc`, `acc`, `afc`, `afd`                                                                                 |
| Input mode     | `text-only`, `audio-only`, `text-audio`                                                                           |
| Model family   | `bilstm`, `bilstm-mfcc`, `bilstm-transformer`, `transformer`, `transformer-encoder`, `mmtransformer`, `csa`, `ensemble`, `multa` |
| Text model     | `bert`, `roberta`                                                                                                 |
| Source         | `mancini-et-al-2022`, `anonymous`                                                                                 |

Tasks that encode audio with a pretrained model come in one variant per model.
The variant adds a tag such as `audio.model_card=facebook/hubert-base-ls960`.
Without that tag, a task uses the card in its registration.
That card is `facebook/wav2vec2-base-960h`, except for the two AFD tasks that use WavLM.

### Add a dataset or a model
Components live in `mamkit/components`, and their configurations live in `mamkit/configurations`.
A new corpus subclasses `Loader` in `mamkit/components/loaders.py`.
A new encoder is a PyTorch module that `Classifier` wraps with a classification head.
A new experiment registers a configuration for each new component and a task that names them by key.
The [Cinnamon documentation](https://nlp.unibo.it/cinnamon/) describes configurations, registrations, and external project directories.

Run `cmn-check -dir mamkit --deep` after editing the registrations.
It reports unresolved keys and constructor arguments that do not match their configuration.

## 🧠 Structure
The `mamkit` package holds four subpackages:

1. `components` holds the loaders, split strategies, featurizers, encoders, models, the Lightning module, and the task.
2. `configurations` holds the configuration classes and every registration, including the experiments of the paper under `configurations/experiments`.
3. `modules` holds the recurrent and attention layers that the encoders use.
4. `utility` holds downloads, audio and text processing, and metrics.

## 📚 Website and Documentation 
The documentation is available [here](https://nlp-unibo.github.io/mamkit/mamkit.html).

The website is available [here](https://nlp-unibo.github.io/mamkit/).

Our website provides a comprehensive overview of the toolkit, including installation instructions, usage examples, and a detailed description of the toolkit's components.
The website also describes the datasets, tasks, and models of the toolkit.
Its leaderboard reports the results of the current models on each dataset.

## 🤝 Contributing 
We welcome contributions to MAMKit!  Please refer to the [contributing guidelines](https://nlp-unibo.github.io/mamkit/contribute.html) for more information.

## 📧 Contact Us
For questions or suggestions, contact us: [Eleonora Mancini](mailto:e.mancini@unibo.it), [Federico Ruggeri](mailto:federico.ruggeri6@unibo.it).

## 📖 Citation
If you use MAMKit in your research, please cite the following paper:
```
@inproceedings{mancini-etal-2024-mamkit,
    title = "{MAMK}it: A Comprehensive Multimodal Argument Mining Toolkit",
    author = "Mancini, Eleonora  and
      Ruggeri, Federico  and
      Colamonaco, Stefano  and
      Zecca, Andrea  and
      Marro, Samuele  and
      Torroni, Paolo",
    editor = "Ajjour, Yamen  and
      Bar-Haim, Roy  and
      El Baff, Roxanne  and
      Liu, Zhexiong  and
      Skitalinskaya, Gabriella",
    booktitle = "Proceedings of the 11th Workshop on Argument Mining (ArgMining 2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.argmining-1.7",
    doi = "10.18653/v1/2024.argmining-1.7",
    pages = "69--82",
}

```

## 🙏 Acknowledgement
This work was partially supported by project FAIR: Future Artificial Intelligence Research (European Commission NextGeneration EU programme, PNRR-M4C2-Investimento 1.3, PE00000013-"FAIR" - Spoke 8).
