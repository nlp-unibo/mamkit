MAMKit
======

A comprehensive toolkit for multimodal argument mining.

Multimodal argument mining (MAM) studies arguments in speech, where a model reads the transcript of a sentence, its audio, or both.
MAMKit ships four corpora of political debates, five tasks, and the models of the MAMKit paper, under one configuration system and one training loop.
Every published experiment is a registered task, so reproducing a result is running one key.

.. code-block:: console

   pip install mamkit

.. code-block:: python

   from pathlib import Path

   import mamkit
   from cinnamon.registry import Registry
   from mamkit.configurations.keys import TOY_TASK

   Registry.build(directory=Path(mamkit.__file__).parent)

   results = Registry.from_key(TOY_TASK, save_path="results", seeds=[0, 1]).run()
   print(results["test"]["f1"])

The corpus, the split strategy, the featurizers, the model, and the metrics of a task are all registration keys.
Swapping the corpus or the model is therefore a change of key rather than a change of code.
A run writes every score per seed and split, and their summary, to ``results.json``.

Where to start
--------------

- :doc:`concepts/argument-mining` describes the tasks, the input modes, and the sentence pairs that MAMKit models.
- :doc:`tutorials/index` holds two walkthroughs. :doc:`tutorials/quickstart` runs a task end to end, and :doc:`tutorials/custom-model` registers a model of your own.
- :doc:`models/index` compares the model families and gives each one a page.
- :doc:`reference/index` describes the corpora, the featurizers, the tasks, the metrics, every registration, and the results of the paper.
- :doc:`project/contributing` describes the environment, the checks, and the release flow for a change to the toolkit itself.

.. toctree::
   :maxdepth: 2
   :caption: Start here
   :hidden:

   concepts/argument-mining
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Models
   :hidden:

   models/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/index

.. toctree::
   :maxdepth: 2
   :caption: Project
   :hidden:

   project/contributing

Citing
------

If you use MAMKit in your research, please cite the following paper:

.. code-block:: bibtex

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
