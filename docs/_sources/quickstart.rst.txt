.. _quickstart:

Quickstart
*********************************************

TODO

MAMKit provides a modular interface for defining datasets or allowing users to load datasets from the literature.

The toolkit provides a modular interface for defining models, allowing users to compose models from pre-defined components or define custom models.
In particular, MAMkit offers a simple method for both defining custom models and leveraging models from the literature.

*********************************************
Loading a Dataset
*********************************************

In the example that follows, illustrates how to load a dataset.
In this case, a dataset is loaded using the `MMUSED` class from `mamkit.data.datasets`, which extends the `Loader` interface and implements specific functionalities for data loading and retrieval.
Users can specify task and input mode (`text-only`, `audio-only`, or `text-audio`) when loading the data, with options to use default splits or load splits from previous works. The example uses splits from `Mancini et al. (2022) <https://aclanthology.org/2022.argmining-1.15)>`_.

The `get_splits` method of the `loader` returns data splits in the form of a `data.datasets.SplitInfo`. The latter wraps split-specific data, each implementing PyTorch's `Dataset` interface and compliant to the specified input modality (i.e., `text-only`).


.. code-block:: python

    from mamkit.data.datasets import UKDebates, InputMode

    loader = UKDebates(
              task_name='asd',
              input_mode=InputMode.TEXT_ONLY,
              base_data_path=base_data_path)


    split_info = loader.get_splits('mancini-et-al-2022')


The `Loader` interface also allows users to integrate methods defining custom splits as follows:

.. code-block:: python

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

*********************************************
Adding a new Dataset
*********************************************

To add a new dataset, users need to create a new class that extends the `Loader` interface and implements the required functionalities for data loading and retrieval.
The new class should be placed in the `mamkit.data.datasets` module.

*********************************************
Loading a Model
*********************************************

The following example demonstrates how to instantiate a model with a configuration found in the literature.
This configuration is identified by a key, `ConfigKey`, containing all the defining information.
The key is used to fetch the precise configuration of the model from the `configs` package.
Subsequently, the model is retrieved from the `models` package and configured with the specific parameters outlined in the configuration.


.. code-block:: python

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



*********************************************
Defining a custom Model
*********************************************

The example below illustrates that defining a custom model is straightforward.
It entails creating the model within the `models` package, specifically by extending either the `AudioOnlyModel`, `TextOnlyModel`, or `TextAudioModel` classes in the `models.audio`, `models.text`, or `models.text_audio` modules, respectively, depending on the input modality handled by the model.

.. code-block:: python

    class Transformer(TextOnlyModel):

        def __init__(
                self,
                model_card,
                head,
                dropout_rate=0.0,
                is_transformer_trainable: bool = False,
        ): ...

.. code-block:: python

    from mamkit.models.text import Transformer

    model = Transformer(
              model_card='bert-base-uncased',
              dropout_rate=0.1, ...)

*********************************************
Training a Model
*********************************************

Our models are designed to be encapsulated into a PyTorch `LightningModule`, which can be trained using PyTorch Lightning's `Trainer` class.
The following example demonstrates how to wrap and train a model using PyTorch Lightning.

.. code-block:: python

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

*********************************************
Benchmarking
*********************************************

The `mamkit.configs` package simplifies reproducing literature results in a structured manner.
Upon loading the dataset, experiment-specific configurations can be easily retrieved via a configuration key.
This enables instantiating a processor using the same features processor employed in the experiment.

In the example below, we adopt a configuration akin to `Mancini et al. (2022) <https://aclanthology.org/2022.argmining-1.15>`_, employing a BiLSTM model with audio encoded with MFCCs features. Hence, we define a `MFCCExtractor` processor using configuration parameters.

.. code-block:: python

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
