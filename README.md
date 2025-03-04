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
- [Installation](#installation)
- [Usage](#usage)
  - [Data](#data)
    - [Load a Dataset](#load-a-dataset)
    - [Add a New Dataset](#add-a-new-dataset)
  - [Modelling](#modelling)
    - [Load a Model](#load-a-model)
    - [Custom Model Definition](#custom-model-definition)
    
  - [Training](#training)
  - [Benchmarking](#benchmarking)
- [Structure](#structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Contact Us](#contact-us)
- [Citation](#citation)

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
| BiLSTM     | GloVe + BiLSTM    | (Wav2Vec2 &or; MFCCs) + BiLSTM                | Conat-Late   |
| MM-BERT    | BERT              | (Wav2Vec2 &or; HuBERT &or; WavLM) + BiLSTM      | Concat-Late  |
| MM-RoBERTa | RoBERTa           | (Wav2Vec2 &or; HuBERT &or; WavLM) + BiLSTM      | Concat-Late  |
| CSA        | BERT              | (Wav2Vec2 &or; HuBERT &or; WavLM) + Transformer | Concat-Early |
| Ensemble   | BERT              | (Wav2Vec2 &or; HuBERT &or; WavLM) + Transformer | Avg-Late     |
| Mul_TA     | BERT              | (Wav2Vec2 &or; HuBERT &or; WavLM) + Transformer | Cross        |


## 🔧 Installation


### ⚠️ Prerequisites
Before installing MAMKit, ensure you have the following:

- **Python 3.10** (MAMKit is tested with this version)
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

1. Clone the repository and install the requirements:
```bash
git clone git@github.com:nlp-unibo/mamkit.git
cd mamkit
pip install -r requirements.txt
```
2. Access MAMKit in your Python code: 
```bash
import mamkit 
```


## ⚙️ Usage
### Data 
MAMKit provides a modular interface for defining datasets or allowing users to load datasets from the literature. 

#### Load a Dataset
In the example that follows, illustrates how to load a dataset.
In this case, a dataset is loaded using the `MMUSED` class from `mamkit.data.datasets`, which extends the `Loader` interface and implements specific functionalities for data loading and retrieval.
Users can specify task and input mode (`text-only`, `audio-only`, or `text-audio`) when loading the data, with options to use default splits or load splits from previous works. The example uses splits from [Mancini et al. (2022)](https://aclanthology.org/2022.argmining-1.15).

The `get_splits` method of the `loader` returns data splits in the form of a `data.datasets.SplitInfo`. The latter wraps split-specific data, each implementing PyTorch's `Dataset` interface and compliant to the specified input modality (i.e., `text-only`).



```python
from mamkit.data.datasets import UKDebates, InputMode

loader = UKDebates(
          task_name='asd',
          input_mode=InputMode.TEXT_ONLY,
          base_data_path=base_data_path)


split_info = loader.get_splits('mancini-et-al-2022')
```
The `Loader` interface also allows users to integrate methods defining custom splits as follows:

```python
from mamkit.data.datasets import SplitInfo

def custom_splits(self) -> List[SplitInfo]:
    train_df = self.data.iloc[:50]
    val_df = self.data.iloc[50:100]
    test_df = self.data.iloc[100:]
    fold_info = self.build_info_from_splits(train_df=...)
    return [fold_info]
              
loader.add_splits(method=custom_splits,
                  key='custom')

split_info = loader.get_splits('custom')
```

#### Add a New Dataset
To add a new dataset, users need to create a new class that extends the `Loader` interface and implements the required functionalities for data loading and retrieval.
The new class should be placed in the `mamkit.data.datasets` module.

### Modelling
The toolkit provides a modular interface for defining models, allowing users to compose models from pre-defined components or define custom models.
In particular, MAMkit offers a simple method for both defining custom models and leveraging models from the literature. 

#### Load a Model
The following example demonstrates how to instantiate a model with a configuration found in the literature.
This configuration is identified by a key, `ConfigKey`, containing all the defining information.
The key is used to fetch the precise configuration of the model from the `configs` package.
Subsequently, the model is retrieved from the `models` package and configured with the specific parameters outlined in the configuration.
```python
from mamkit.configs.base import ConfigKey
from mamkit.configs.text import TransformerConfig
from mamkit.data.datasets import InputMode

config_key = ConfigKey(
              dataset='mmused', 
              task_name='asd',
              input_mode=InputMode.TEXT_ONLY,
              tags={'mancini-et-al-2022'})

config = TransformerConfig.from_config(
                           key=config_key)
    
model = Transformer(
         model_card=config.model_card,
         dropout_rate=config.dropout_rate
         ...)
```

#### Custom Model Definition 
The example below illustrates that defining a custom model is straightforward. It entails creating the model within the `models` package, specifically by extending either the `AudioOnlyModel`, `TextOnlyModel`, or `TextAudioModel` classes in the `models.audio`, `models.text`, or `models.text_audio` modules, respectively, depending on the input modality handled by the model.
```python
class Transformer(TextOnlyModel):

    def __init__(
            self,
            model_card,
            head,
            dropout_rate=0.0,
            is_transformer_trainable: bool = False,
    ): ...

```

```python
from mamkit.models.text import Transformer

model = Transformer(
          model_card='bert-base-uncased',
          dropout_rate=0.1, ...)
```


### Training
Our models are designed to be encapsulated into a PyTorch `LightningModule`, which can be trained using PyTorch Lightning's `Trainer` class.
The following example demonstrates how to wrap and train a model using PyTorch Lightning. 

```python
from mamkit.utility.model import to_lighting_model
import lightning

model = to_lighting_model(model=model, 
        num_classes=config.num_classes,
        loss_function=...,
        optimizer_class=...)

trainer = lightning.Trainer(max_epochs=100,
                            accelerator='gpu',
                            ...)
trainer.fit(model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)
```

### Benchmarking
The `mamkit.configs` package simplifies reproducing literature results in a structured manner. 
Upon loading the dataset, experiment-specific configurations can be easily retrieved via a configuration key.
This enables instantiating a processor using the same features processor employed in the experiment. 

In the example below, we adopt a configuration akin to [Mancini et al. (2022)](https://aclanthology.org/2022.argmining-1.15), employing a BiLSTM model with audio encoded with MFCCs features. Hence, we define a `MFCCExtractor` processor using configuration parameters.

```python
from mamkit.configs.audio import BiLSTMMFCCsConfig
from mamkit.configs.base import ConfigKey
from mamkit.data.datasets import UKDebates, InputMode
from mamkit.data.processing import MFCCExtractor, UnimodalProcessor
from mamkit.models.audio import BiLSTM

loader = UKDebates(task_name='asd',
           input_mode=InputMode.AUDIO_ONLY)

config = BiLSTMMFCCsConfig.from_config(
                key=ConfigKey(dataset='ukdebates',
                input_mode=InputMode.AUDIO_ONLY,
                task_name='asd',
                tags='mancini-et-al-2022'))


for split_info in loader.get_splits(
                         key='mancini-et-al-2022'):
    processor = 
        UnimodalProcessor(
            features_processor=MFCCExtractor(
                mfccs=config.mfccs, ...))

    split_info.train = processor(split_info.train)
    ...
    model = BiLSTM(embedding_dim=
                    config.embedding_dim, ...)
```

## 🧠 Structure
The toolkit is organized into five main components: `configs`, `data`, `models`,  `modules` and `utility`. 
In addition to that, the toolkit provides a `demos` directory for running all the experiments presented in the paper.
The figure below illustrates the toolkit's structure.

![Toolkit Structure](figures/mamkit2.png)


## 📚 Website and Documentation 
The documentation is available [here](https://nlp-unibo.github.io/mamkit/mamkit.html).

The website is available [here](https://nlp-unibo.github.io/mamkit/).

Our website provides a comprehensive overview of the toolkit, including installation instructions, usage examples, and a detailed description of the toolkit's components.
Moreover, the website provides a detailed description of the datasets, tasks, and models available in the toolkit, together with a leaderboard of the results obtained on the datasets with the current models.

## 🤝 Contributing 
We welcome contributions to MAMKit!  Please refer to the [contributing guidelines](https://nlp-unibo.github.io/mamkit/contribute.html) for more information.

## 📧 Contact Us
For any questions or suggestions, don't hesitate to contact  us: [Eleonora Mancini](mailto:e.mancini@unibo.it), [Federico Ruggeri](mailto:federico.ruggeri6@unibo.it).

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
