Configurations
==============

``mamkit.configurations.keys`` holds every ``RegistrationKey`` that the configuration modules register, so a script imports a key from one place.
The experiments of the MAMKit paper are registered in ``mamkit.configurations.experiments``, one module per corpus.

Every registration
------------------

The cards below are built from the registry once per documentation build, so every key shown exists and every default is the one a run gets.
Each card shows the registration name, its tags, the component it builds, and every parameter with its type and default.
The search box filters by any of those.
For example, ``ukdebates bilstm`` narrows to the UKDebates BiLSTM registrations, and ``loader`` narrows to the corpora.

.. registry-cards::

Reading a card
--------------

A key is a name and a set of tags in a namespace, and ``Registry.from_key`` resolves the key to a component.
A parameter whose default reads as ``name[tag, tag]`` is itself a registration key, which is how one configuration names another.
A card marked **runnable** carries a ``run_method``, so ``cmn-run`` offers it from the command line.

A key that a variant derives carries one tag per varied field, such as ``audio.model_card=facebook/hubert-base-ls960``.
Each derived key has a card of its own.
Writing registrations of your own, in a package beside MAMKit, is :doc:`../tutorials/custom-model`.

API
---

.. automodule:: mamkit.configurations.keys
   :members:

.. automodule:: mamkit.configurations.datasets
   :members:

.. automodule:: mamkit.configurations.splits
   :members:

.. automodule:: mamkit.configurations.features
   :members:

.. automodule:: mamkit.configurations.models
   :members:

.. automodule:: mamkit.configurations.metrics
   :members:

.. automodule:: mamkit.configurations.tasks
   :members:
