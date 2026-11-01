.. mamkit documentation master file, created by
   sphinx-quickstart on Sun May 21 18:00:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MAMKit
================================================

MAMKit is an open-source, publicly available PyTorch toolkit designed to access and develop datasets, models, and benchmarks for Multimodal Argument Mining (MAM). It provides a flexible interface for accessing and integrating datasets, models, and preprocessing strategies through composition or custom definition. MAMKit is designed to be extendible, ensure replicability, and provide a shared interface as a common foundation for experimentation in the field.

MAMKit is built on `Cinnamon <https://nlp.unibo.it/cinnamon/>`_.
Each corpus, split strategy, featurizer, model, and metric is a component with a registered configuration.
Each experiment of the paper is a registered task that combines them.

Structure
================================================

The ``mamkit`` package holds four subpackages:

1. ``components`` holds the loaders, split strategies, featurizers, encoders, models, the Lightning module, and the task.
2. ``configurations`` holds the configuration classes and every registration, including the experiments of the paper under ``configurations/experiments``.
3. ``modules`` holds the recurrent and attention layers that the encoders use.
4. ``utility`` holds downloads, audio and text processing, and metrics.

Prerequisites
=============

Before installing MAMKit, ensure you have the following:

- **Python 3.10 to 3.13** (MAMKit is tested with these versions)
- **FFmpeg** (Required for audio processing)  

  You can install it via:

  .. code-block:: bash

     sudo apt install ffmpeg  # Debian/Ubuntu  
     brew install ffmpeg      # macOS  
     choco install ffmpeg     # Windows (using Chocolatey)  

  For other installation methods, refer to the `FFmpeg official website <https://www.ffmpeg.org/>`_.

Install via PyPi
================

1. Install MAMKit using PyPi:

   .. code-block:: bash

      pip install mamkit  

2. Access MAMKit in your Python code:

   .. code-block:: python

      import mamkit  

Install from GitHub
===================

This installation is recommended for users who wish to conduct experiments and customize the toolkit according to their needs.

1. Clone the repository and install it in editable mode:

   .. code-block:: bash

      git clone git@github.com:nlp-unibo/mamkit.git
      cd mamkit
      pip install -e .  

2. Access MAMKit in your Python code:

   .. code-block:: python

      import mamkit  



Contribute
===============================================

Feel free to submit a pull request!
We welcome new datasets, models, and any other contribution that can improve the toolkit!

MAMKit is meant to be a community project :)


Contact
===============================================

Don't hesitate to contact:
- `Eleonora Mancini <e.mancini@unibo.it>`_
- `Federico Ruggeri <federico.ruggeri6@unibo.it>`_

for questions/doubts/issues!


Citing
===============================================

If you use MAMKit in your research, please cite the following paper:

.. code-block::

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


.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Contents:
   :titlesonly:

   Quick Start <quickstart.rst>
   Leaderboard <leaderboard.rst>
   Datasets <datasets.rst>
   Models <models.rst>
   Contribute <contribute.rst>
   mamkit <mamkit.rst>