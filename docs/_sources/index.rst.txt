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

The toolkit is organized into five main components: `configs`, `data`, `models`,  `modules` and `utility`.
In addition to that, the toolkit provides a `demos` directory for running all the experiments presented in the paper.
The figure below illustrates the toolkit's structure.

.. image:: img/mamkit2.png
    :scale: 60%
    :align: center


Install
===============================================

pip
   .. code-block:: bash

      pip install mamkit

git
   .. code-block:: bash

      git clone https://github.com/lt-nlp-lab-unibo/mamkit


===============================================
Contribute
===============================================

Feel free to submit a merge request!

MAMKit is meant to be a community project :)

===============================================
Contact
===============================================

Don't hesitate to contact:
- `Eleonora Mancini <e.mancini@unibo.it>`_
- `Federico Ruggeri <federico.ruggeri6@unibo.it>`_

for questions/doubts/issues!

===============================================
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