.. mamkit documentation master file, created by
   sphinx-quickstart on Sun May 21 18:00:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MAMKit
================================================

MAMKit is an open-source, publicly available PyTorch toolkit designed to access and develop datasets, models, and benchmarks for Multimodal Argument Mining (MAM). It provides a flexible interface for accessing and integrating datasets, models, and preprocessing strategies through composition or custom definition. MAMKit is designed to be extendible, ensure replicability, and provide a shared interface as a common foundation for experimentation in the field.

Currently, MAMKit offers 4 datasets, 4 tasks and 6 distinct model architectures, along with audio and text processing capabilities, organized in 5 main components.

Structure
================================================

The toolkit is organized into five main components: ``configs``, ``data``, ``models``,  ``modules`` and ``utility``.
In addition to that, the toolkit provides a ``demos`` directory for running all the experiments presented in the paper.
The figure below illustrates the toolkit's structure.

.. image:: img/mamkit2-resized.png
    :align: center



Prerequisites
=============

Before installing MAMKit, ensure you have the following:

- **Python 3.10** (MAMKit is tested with this version)
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

1. Clone the repository and install the requirements:

   .. code-block:: bash

      git clone git@github.com:nlp-unibo/mamkit.git
      cd mamkit
      pip install -r requirements.txt  

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

   @inproceedings{TBAmamkit,
     title={MAMKit: A Comprehensive Multimodal Argument Mining Toolkit},
     author={TBA},
     booktitle={TBA},
     year={TBA}
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