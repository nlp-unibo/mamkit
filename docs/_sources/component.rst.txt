.. _component:

Component
*********************************************

A ``Component`` generally receive data and produce other data as output: i.e., a data transformation process.

.. image:: img/component.png
    :scale: 70%
    :align: center

Cinnamon has minimal APIs for ``Component`` to allow user customization.

*********************************************
Run
*********************************************

The ``run()`` function is the general entrypoint for ``Component``.

A general-purpose function offers a simple interface for calling ``Component`` without any particular requirement.

However, using the ``run()`` function is not mandatory. Users can explicitly call ``Component`` as they wish.

*********************************************
Save & Load
*********************************************

The internal state of a ``Component`` (e.g., its ``Configuration``) can be stored and loaded.

Cinnamon offers ``save()`` and ``load()`` functionalities, respectively.

.. note::
    ``save()`` and ``load()`` support nesting! Thus, nested ``Configuration`` and ``Component`` can be loaded in one step.

Selecting which information to store is up to the user: cinnamon offers a ``prepare_save_data()`` method that defines a dictionary of field-value pairs to store.

*********************************************
Find
*********************************************

A handy utility function of ``Component`` is the ``find()`` method.

In many cases, one might need to retrieve a particular ``Parameter`` value (especially in nested ``Component``).

.. code-block:: python

    samples_amount = component.find('samples_amount')

The ``find()`` searches over the ``Component`` recursively and returns the **first** hit.