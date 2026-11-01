Metrics
=======

Each corpus registers one F1 score, under the name ``f1``.
The metrics score the class with the highest logit, on the validation set and on the test set.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Corpus
     - Tasks
     - Score
   * - UKDebates
     - ASD
     - F1 of the claim class, from ``torchmetrics.classification.BinaryF1Score``
   * - M-Arg
     - ARC
     - F1 of the support and attack classes, averaged, from ``ClassSubsetMulticlassF1Score``
   * - MM-USED
     - ASD, ACC
     - Macro F1 over the two classes, from ``torchmetrics.F1Score``
   * - MM-USED-fallacy
     - AFC, AFD
     - Macro F1 over the six fallacy classes for AFC, and over the two classes for AFD, from ``torchmetrics.F1Score``

The M-Arg score leaves out the ``neither`` class.

API
---

.. automodule:: mamkit.utility.metrics
   :members:
