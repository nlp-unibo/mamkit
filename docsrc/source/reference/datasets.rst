Datasets
========

A loader downloads a corpus into a folder of the working directory, or of ``base_data_path``, on first use.
It reads the table of one task and one input mode, and converts row subsets into a ``MAMDataset``.
A split strategy decides which rows form the training, validation, and test sets.

Corpora
-------

.. list-table::
   :header-rows: 1
   :widths: 18 20 12 50

   * - Corpus
     - Loader
     - Tasks
     - Description
   * - UKDebates
     - ``UKDebates``
     - ASD
     - Transcriptions and audio of the three candidates for the 2015 UK Prime Ministerial elections, in a two-hour debate aired by Sky News: David Cameron, Nick Clegg, and Ed Miliband. It contains 386 sentences and their audio samples.
   * - M-Arg
     - ``MArg``
     - ARC
     - Transcriptions and audio of four candidates and a debate moderator in the 2020 US presidential debates, over 18 topics. Crowd workers labelled 4,104 sentence pairs as support, attack, or neither. The subset with agreement confidence of at least 85%, 2,443 pairs, is the one commonly used for evaluation.
   * - MM-USED
     - ``MMUSED``
     - ASD, ACC
     - Transcripts and audio of US presidential debates aired from 1960 to 2016. It contains 23,505 labelled sentences [#alignment]_ from 39 debates and 26 speakers.
   * - MM-USED-fallacy
     - ``MMUSEDFallacy``
     - AFC, AFD
     - Fallacious snippets from US presidential debates aired from 1960 to 2016. It contains 1,278 samples [#alignment]_ labelled with six fallacy categories.

.. [#alignment] The alignment of these corpora was refined after their publication, so their sample counts differ from the original papers.

Each loader takes a ``task_name``, which must be one of its tasks.
It also takes an ``input_mode``, which is ``text-only``, ``audio-only``, or ``text-audio``.
``MArg`` takes the annotation ``confidence`` of its pairs, 0.85 or 1.0.
It cuts each clip with ``clip_buffer`` seconds before and after the sentence, 2 by default, at ``sample_rate``, or at the rate of the recording by default.
Each clip setting is built in a folder of its own.
``MMUSED`` and ``MMUSEDFallacy`` cut their clips at ``sample_rate``, 16 kHz by default.
``Toy`` is a synthetic text corpus for smoke tests, which needs no download.

Split strategies
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Strategy
     - Splits
   * - ``FoldFiles``
     - The folds of a JSON file of row positions, downloaded once. UKDebates and M-Arg use the folds of Mancini et al. (2022).
   * - ``ColumnSplit``
     - One split read from a column that names the set of each row. MM-USED uses it.
   * - ``LeaveOneDialogueOut``
     - One fold per dialogue, which is the test set of that fold, with validation dialogues drawn from the others. MM-USED-fallacy uses the folds of Mancini et al. (2024).
   * - ``DialogueHoldout``
     - One split that tests on a fixed list of dialogues, the split of the MM-ArgFallacy 2025 shared task.
   * - ``AllTrain``
     - One split that trains on every row, without validation or test sets.

Samples
-------

A ``MAMDataset`` holds one list of inputs per modality, and one label per sample.
A pair dataset holds such a dictionary for side ``a`` and side ``b``.
An optional ``context`` has the same layout as the inputs.

API
---

.. automodule:: mamkit.components.loaders
   :members:

.. automodule:: mamkit.components.splits
   :members:

.. automodule:: mamkit.components.datasets
   :members:
