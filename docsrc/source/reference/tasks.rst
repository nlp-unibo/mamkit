Tasks
=====

A task is one experiment from start to finish: a corpus, a split strategy, a featurizer per modality, a model, its metrics, and a list of seeds.
``Task`` is registered with ``run_method="run"``, so ``cmn-run`` drives it from the command line.

.. code-block:: python

   from pathlib import Path

   import mamkit
   from cinnamon.registry import Registry
   from mamkit.configurations.keys import TOY_TASK

   Registry.build(directory=Path(mamkit.__file__).parent)
   results = Registry.from_key(TOY_TASK, seeds=[42, 1337]).run()

What a run does
---------------

``run`` builds the loader and the split strategy once.
Then, for each seed, in order:

1. Seed Python, NumPy, and PyTorch through Lightning, and compute the splits.
2. For each split, build fresh featurizers and a fresh model from their keys.
3. Fit the featurizers on the training set, transform every set, and pass the fitted vocabulary to every module that defines ``load_vocabulary``.
4. Train with early stopping on the validation loss, with ``patience`` epochs, and keep the checkpoint with the lowest validation loss.
5. Score that checkpoint on the validation set and on the test set.

A split without a validation set trains for ``max_epochs`` and scores the last weights.
A split without a test set is not scored on it.

``trainer_args`` go to the Lightning ``Trainer`` unchanged.
The registered experiments set ``accelerator`` to ``auto``, ``devices`` to 1, ``accumulate_grad_batches`` to 3, and ``max_epochs`` to the value of their source.

What a run writes
-----------------

``run`` returns the results, and with ``save_path`` it also writes them to ``results.json`` there:

.. code-block:: text

   {
     "runs": [{"seed": 42, "split": 0, "validation": {...}, "test": {...}}, ...],
     "test": {
       "f1": {"per_seed_mean": [...], "per_seed_std": [...], "mean": ..., "std": ...}
     },
     "validation": {...}
   }

``runs`` holds every score of every seed and split.
For each set and metric, ``per_seed_mean`` and ``per_seed_std`` summarize the splits of each seed, and ``mean`` and ``std`` summarize the per-seed means.
Lightning writes the checkpoints and its logs under ``save_path`` as well.

API
---

.. automodule:: mamkit.components.tasks
   :members:

.. automodule:: mamkit.components.lightning
   :members:
