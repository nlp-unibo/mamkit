Quickstart
==========

This tutorial runs a task on the synthetic ``toy`` corpus, which needs no download and trains in seconds on a CPU.
It then runs an experiment of the MAMKit paper.

.. code-block:: console

   pip install mamkit

Audio tasks also need `FFmpeg <https://www.ffmpeg.org/>`_ on the system path.

1. Build the registry
---------------------

Nothing is registered until something asks.
One call executes every registration that MAMKit ships and reports which keys passed validation:

.. code-block:: python

   from pathlib import Path

   import mamkit
   from cinnamon.registry import Registry

   valid, invalid = Registry.build(directory=Path(mamkit.__file__).parent)

``invalid`` is empty for the toolkit itself.

2. Run a task
-------------

A task is one experiment from start to finish.
It names a corpus, a split strategy, a featurizer per modality, a model, its metrics, and a list of seeds, each by key.

.. code-block:: python

   from mamkit.configurations.keys import TOY_TASK

   task = Registry.from_key(TOY_TASK, save_path="results", seeds=[0, 1])
   results = task.run()

The keyword arguments override fields of the registered configuration for this run only.
For each seed and each split, the task builds fresh featurizers and a fresh model.
It fits the featurizers on the training set and trains with early stopping on the validation loss.
It then scores the best checkpoint on the validation and test sets.

``results`` holds every score of every run, and a summary per set and metric:

.. code-block:: python

   >>> results["test"]["f1"]
   {'per_seed_mean': [1.0, 1.0], 'per_seed_std': [0.0, 0.0], 'mean': 1.0, 'std': 0.0}

``per_seed_mean`` and ``per_seed_std`` summarize the splits of each seed.
``mean`` and ``std`` summarize the per-seed means.
The toy corpus marks each label with a cue word, so a perfect score only shows that the run holds together.

The same results are written to disk under ``save_path``:

.. code-block:: text

   results/
   ├── results.json
   └── checkpoints/
       ├── epoch=9-step=150.ckpt
       └── epoch=9-step=150-v1.ckpt

3. Run an experiment of the paper
---------------------------------

Every experiment of the MAMKit paper is a task with the name ``task`` in the ``mamkit`` namespace.
Its tags name the corpus, the task, the input mode, the model family, and the source of the configuration:

.. code-block:: python

   from cinnamon.registry import RegistrationKey

   key = RegistrationKey(
       name="task",
       tags={"ukdebates", "asd", "text-only", "bilstm", "mancini-et-al-2022"},
       namespace="mamkit",
   )
   results = Registry.from_key(key).run()

This task downloads UKDebates and GloVe vectors on first use, into the working directory.
It writes its results under ``results/ukdebates/asd/text-only/bilstm/mancini-et-al-2022``.
``Registry.retrieve_configuration(key)`` returns the configuration without building anything, so its fields can be read first.
:doc:`../reference/configurations` lists every registered key.

4. Run from the command line
----------------------------

``cmn-run`` lists the registered tasks, asks which ones to run, and runs them.
It needs the ``cli`` extra of Cinnamon:

.. code-block:: console

   pip install "cinnamon-core[cli]"
   cmn-run -dir "$(python -c 'import mamkit, os; print(os.path.dirname(mamkit.__file__))')"

The command asks for a namespace, a name, and tags.
Select the ``task`` name, then add tags until the list holds the experiments to run.
From a clone of the repository, ``cmn-run -dir mamkit`` is enough.
