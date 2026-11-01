A model of your own
===================

This tutorial writes a new text encoder, registers it in a package beside MAMKit, and trains it with the toy task of :doc:`quickstart`.
The same steps register a new audio encoder or a new task.

Layout
------

A method of your own lives in a package of its own:

.. code-block:: text

   mymodel/
   ├── __init__.py
   ├── components/
   │   ├── __init__.py
   │   └── encoders.py        # the encoder
   └── configurations/
       ├── __init__.py
       ├── keys.py            # the keys, in a namespace of your own
       └── registrations.py   # what each key builds

The registry executes only the Python files under a directory named ``configurations``.
A registration written anywhere else never runs, so the key it declares does not exist.
Components can live anywhere, because a registration names them by import path.

The encoder
-----------

An encoder is a ``torch.nn.Module`` that reads one batch of samples and returns one vector per sample.
A sample maps each modality to its fields, so the encoder below reads ``sample["text"]``.
Its ``output_dim`` attribute tells ``Classifier`` how wide the vector is.

.. code-block:: python

   # mymodel/components/encoders.py
   import torch as th

   from mamkit.components.encoders import masked_mean


   class TextMeanPool(th.nn.Module):
       """Averages token embeddings over the valid tokens of each text."""

       def __init__(self, embedding_dim: int):
           super().__init__()
           self.embedding_dim = embedding_dim
           self.embedding = None
           self.output_dim = embedding_dim

       def load_vocabulary(self, vocab_size, embedding_matrix=None):
           self.embedding = th.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
           if embedding_matrix is not None:
               self.embedding.weight.data = embedding_matrix

       def forward(self, sample: dict) -> th.Tensor:
           text = sample["text"]
           return masked_mean(self.embedding(text["inputs"]), text["input_mask"])

The vocabulary is fitted on the training set of each split, so its size is unknown when the encoder is built.
``Task`` therefore calls ``load_vocabulary`` on every module that defines it, with the size and the optional GloVe matrix that ``VocabularyText`` computed.
An encoder over pretrained features does not need this method.

The keys
--------

Every key the package registers is declared once, in ``keys.py``, from a ``Namespace`` of its own:

.. code-block:: python

   # mymodel/configurations/keys.py
   from cinnamon.registry import Namespace

   NAMESPACE = Namespace("mymodel")

   MEAN_POOL = NAMESPACE.key("encoder", "mean-pool")
   MEAN_POOL_CLASSIFIER = NAMESPACE.key("model", "classifier", "mean-pool")
   MEAN_POOL_TASK = NAMESPACE.key("task", "toy", "mean-pool")

The registrations
-----------------

A configuration holds the constructor arguments of a component, and a registration binds it to a key and a component path.
The classifier and the task reuse the configurations of MAMKit and change one field each:

.. code-block:: python

   # mymodel/configurations/registrations.py
   from typing import List

   from cinnamon.configuration import Configuration, Param
   from cinnamon.registry import RegistrationKey, register_class

   from mamkit.configurations.models import ClassifierConfig
   from mamkit.configurations.tasks import ToyTaskConfig
   from mymodel.configurations.keys import MEAN_POOL, MEAN_POOL_CLASSIFIER, MEAN_POOL_TASK


   @register_class(key=MEAN_POOL, component="mymodel.components.encoders.TextMeanPool")
   class TextMeanPoolConfig(Configuration):
       embedding_dim: int = Param(16, ge=1)


   @register_class(
       key=MEAN_POOL_CLASSIFIER, component="mamkit.components.models.Classifier"
   )
   class MeanPoolClassifierConfig(ClassifierConfig):
       encoder: RegistrationKey = Param(MEAN_POOL)
       head: List[int] = Param([2])


   @register_class(
       key=MEAN_POOL_TASK, component="mamkit.components.tasks.Task", run_method="run"
   )
   class MeanPoolTaskConfig(ToyTaskConfig):
       model: RegistrationKey = Param(MEAN_POOL_CLASSIFIER)

``run_method="run"`` makes the task runnable from ``cmn-run``.

Run it
------

The registry builds your package with MAMKit beside it:

.. code-block:: python

   from pathlib import Path

   import mamkit
   import mymodel
   from cinnamon.registry import Registry
   from mymodel.configurations.keys import MEAN_POOL_TASK

   valid, invalid = Registry.build(
       directory=Path(mymodel.__file__).parent,
       external_directories=[Path(mamkit.__file__).parent],
   )
   results = Registry.from_key(MEAN_POOL_TASK, save_path="results").run()

``directory`` is the package that runs, and ``external_directories`` are indexed so that their keys can be referenced.
Building MAMKit alone leaves your keys absent.
``cmn-check -dir mymodel -ext externals.json --deep`` reports unresolved keys and constructor arguments that do not match their configuration.
``externals.json`` is a JSON list that holds the path of the MAMKit package directory.

To run the encoder on a corpus of the paper, register a task whose ``loader``, ``split``, ``text``, and ``metrics`` fields name the keys of that corpus.
:doc:`../reference/configurations` lists them.
