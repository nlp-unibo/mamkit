.. _quickstart:

Quickstart
*********************************************

MAMKit is built on `Cinnamon <https://nlp.unibo.it/cinnamon/>`_.
Each corpus, split strategy, featurizer, model, and metric is a component with a registered configuration.
Each experiment of the paper is a registered task that combines them.
A task trains a fresh model on every split of its corpus, once per seed, and reports the mean and standard deviation of every metric.

*********************************************
Running a registered experiment
*********************************************

The ``cmn-run`` command lists the registered tasks, asks which ones to run, and runs them.
It needs the ``cli`` extra of Cinnamon:

.. code-block:: bash

    pip install "cinnamon-core[cli]"

From a clone of the repository, point ``cmn-run`` to the package directory:

.. code-block:: bash

    cmn-run -dir mamkit

From a PyPI installation, point it to the installed package:

.. code-block:: bash

    cmn-run -dir "$(python -c 'import mamkit, os; print(os.path.dirname(mamkit.__file__))')"

The command asks for a namespace, a name, and tags.
Select the ``task`` name, then add tags until the list holds the experiments to run.
Each task writes ``results.json`` under its ``save_path``, which is relative to the working directory.
For example, the BiLSTM of Mancini et al. (2022) on UKDebates writes to ``results/ukdebates/asd/text-only/bilstm/mancini-et-al-2022``.
On first use, each corpus is downloaded into a folder of the working directory, such as ``UKDebates/``.

*********************************************
Running an experiment from Python
*********************************************

A task can also be built and run from a script:

.. code-block:: python

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

``Registry.retrieve_configuration(key)`` returns the configuration without building the task, so its fields can be inspected or changed first.

*********************************************
Selecting experiments by tag
*********************************************

Every task key has the name ``task`` and the namespace ``mamkit``.
Its tags name the corpus, the task, the input mode, the model family, and the source of the configuration.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Field**
     - **Tags**
   * - Corpus
     - ``ukdebates``, ``marg``, ``mmused``, ``mmused-fallacy``
   * - Task
     - ``asd``, ``arc``, ``acc``, ``afc``, ``afd``
   * - Input mode
     - ``text-only``, ``audio-only``, ``text-audio``
   * - Model family
     - ``bilstm``, ``bilstm-mfcc``, ``bilstm-transformer``, ``transformer``, ``transformer-encoder``, ``mmtransformer``, ``csa``, ``ensemble``, ``multa``
   * - Text model
     - ``bert``, ``roberta``
   * - Source
     - ``mancini-et-al-2022``, ``anonymous``

Tasks that encode audio with a pretrained model come in one variant per model.
The variant adds a tag such as ``audio.model_card=facebook/hubert-base-ls960``.
Without that tag, a task uses the card in its registration.
That card is ``facebook/wav2vec2-base-960h``, except for the two AFD tasks that use WavLM.

*********************************************
Adding a dataset or a model
*********************************************

Components live in ``mamkit/components``, and their configurations live in ``mamkit/configurations``.
A new corpus subclasses ``Loader`` in ``mamkit/components/loaders.py``.
A new encoder is a PyTorch module that ``Classifier`` wraps with a classification head.
A new experiment registers a configuration for each new component and a task that names them by key.
The `Cinnamon documentation <https://nlp.unibo.it/cinnamon/>`_ describes configurations, registrations, and external project directories.

Run ``cmn-check -dir mamkit --deep`` after editing the registrations.
It reports unresolved keys and constructor arguments that do not match their configuration.
