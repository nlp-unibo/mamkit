.. _configuration:

Configuration
*********************************************

A ``Configuration`` stores all the hyper-parameters of a ``Component``.

.. image:: img/configuration.png
    :scale: 70%
    :align: center

Each hyper-parameter is wrapped into a ``Parameter``.

.. image:: img/configuration_params.png
    :scale: 60%
    :align: center

-----------------------------------------------
Parameter
-----------------------------------------------

A ``Parameter`` is a useful wrapper for storing additional metadata like

- type hints
- textual descriptions
- allowed value range
- possible value variants
- optional tags for efficient ``Parameter`` search

For instance, the following code defines an integer ``Parameter`` with a specific allowed value range

.. code-block:: python

    x = Parameter(name='x', value=5, type_hint=int, allowed_range=lambda p: p in [1, 5, 10])


The ``name`` field is the ``Parameter`` unique identifier, just like a class field name.

The ``type_hint`` field inherently introduces a condition such that ``x`` can only store an integer value.

The ``allowed_range`` field also introduces a condition such that ``x`` can only be set to ``1,5`` or ``10``.

In a ``Configuration``, the above code becomes

.. code-block:: python

    config.add(name='x', value=5, type_hint=int, allowed_range=lambda p: p in [1, 5, 10])

A ``Configuration`` is built just like a python dictionary. Thus, we can also instantiate a ``Configuration`` from a dictionary as follows

.. code-block:: python

    config = Configuration({'x': 10})

.. note::

    Note that the above method is not recommended since only ``name`` and ``value`` fields can be set.

*********************************************
Adding conditions
*********************************************

Consider our data loader recurring example, we can define a more advanced ``DataLoaderConfig`` as follows:

.. code-block:: python

    class DataLoaderConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='samples_amount',
                       type_hint=int,
                       description="Number of samples to load")
            config.add(name='name',
                       type_hint=str,
                       description="Unique dataset identifier",
                       is_required=True)
            config.add(name='has_test_split_only',
                       value=False,
                       type_hint=bool,
                       description="Whether DataLoader has test split only or not")
            config.add(name='has_val_split',
                       value=True,
                       type_hint=bool,
                       description="Whether DataLoader has a val split or not")
            config.add(name='has_test_split',
                       value=True,
                       type_hint=bool,
                       description="Whether DataLoader has a test split or not")

            return config


Moreover, we can add some **conditions** as well

.. code-block:: python

    class DataLoaderConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='samples_amount',
                       type_hint=int,
                       description="Number of samples to load",
                       allowed_range=lambda p: p > 0)
            config.add(name='name',
                       type_hint=str,
                       description="Unique dataset identifier",
                       is_required=True)
            config.add(name='has_val_split',
                       value=True,
                       type_hint=bool,
                       description="Whether DataLoader has a val split or not")
            config.add(name='has_test_split',
                       value=True,
                       type_hint=bool,
                       description="Whether DataLoader has a test split or not")

            config.add_condition(name='at_least_one_split',
                                 condition=lambda c: c.has_val_split or c.has_test_split)

            return config

In this example, we have 1 **explicit** condition and 6 **implicit** ones:

* ``samples_amount``: ``type_hint`` and ``allowed_range`` conditions (total: 2 implicit conditions).

* ``name``: ``type_hint`` and ``is_required`` conditions (total: 2 implicit conditions).

* ``has_val_split``: ``type_hint`` condition (total: 1 implicit condition).

* ``has_test_split``: ``type_hint`` condition (total: 1 implicit condition).


*********************************************
Validating a Configuration
*********************************************

All ``Configuration`` conditions are **not** executed **automatically**.

The ``Configuration.validate()`` method runs all conditions in sequence to check if the ``Configuration`` can be used.

In cinnamon, the validation of a ``Configuration`` is performed when building a ``Component`` via ``Registry.build_component(...)`` or ``Registry.build_component_from_key(...)``.

*********************************************
Getting a Configuration (delta) copy
*********************************************

In many cases, we may need a slightly modified ``Configuration`` instance.

We can quickly create a ``Configuration`` instance delta copy by only specifying the hyper-parameters to update

.. code-block:: python

    config = DataLoaderConfig.get_default()
    delta_copy = config.get_delta_copy(params={'samples_amount': 500})

We have create a delta copy of ``DataLoaderConfig`` instance that only loads the first 500 samples.

Moreover, we can create delta copy of ``Configuration`` factories as well. This functionality is useful when registering
slightly different variations of a ``Configuration`` template.

.. code-block:: python

    Registry.add_configuration(constructor=DataLoaderConfig.get_delta_class_copy,
                               constructor_kwargs={'params': {'samples_amount': 500}},
                               name=...,
                               tags=...,
                               namespace=...)

We have registered a ``DataLoaderConfig`` template whose instances only load the first 500 samples.


*********************************************
Configuration variants
*********************************************

In a project, we may define a general ``Component`` and many ``Configuration`` bound to it.

In some of these cases, each of these ``Configuration`` are just slight hyper-parameter variations of a single ``Configuration`` template.

In cinnamon, we can avoid explicitly defining all these ``Configuration`` templates and relying on the notion of **configuration variant**.

A configuration variant is a ``Configuration`` template that has at least one different hyper-parameter value.

We define variants by specifying the ``variants`` field when adding a hyper-parameter to the ``Configuration``.

.. code-block:: python

    class MyConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='param_1',
                       value=True,
                       type_hint=bool,
                       variants=[False, True])
            config.add(name='param_2',
                       value=False,
                       type_hint=bool,
                       variants=[False, True])
            return config

In the above code example, ``MyConfig`` has ``param_1`` and ``param_2`` boolean hyper-parameters.
Both of them specify the ``False`` and ``True`` value variants, thus, defining four different ``MyConfig`` templates, one for each combination of the two hyper-parameters.

We can tell the ``Registry`` to keep track of these variants when registering ``MyConfig``:

.. code-block:: python

    Registry.add_and_bind_variants(config_class=MyConfig,
                                   component_class=Component,
                                   name='config',
                                   namespace='showcasing')

The ``Registry`` distinguishes registered variants by adding hyper-parameter name-value pairs as tags (e.g., ``param_1=False``).

*********************************************
Nested Configurations
*********************************************

One core functionality of cinnamon is that ``Configuration`` can be nested to build more sophisticated ones (the same applies for ``Component``).

Consider the following example:

.. code-block:: python

    class ParentConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='param_1',
                       value=True,
                       type_hint=bool,
                       variants=[False, True])
            config.add(name='param_2',
                       value=False,
                       type_hint=bool,
                       variants=[False, True])
            config.add(name='child',
                       value=RegistrationKey(name='config_a',
                                             namespace='testing'),
                       is_child=True)
            return config


    class NestedChild(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='child',
                       value=RegistrationKey(name='config_c',
                                             namespace='testing'),
                       is_child=True)

            return config

    @register
    def register_configurations():
        Registry.add_and_bind(config_class=NestedChild,
                               component_class=Component,
                               name='config_a',
                               namespace='testing')
        Registry.add_and_bind(config_class=Configuration,
                                   component_class=Component,
                                   name='config_c',
                                   namespace='testing')

In the above example, ``ParentConfig`` has a child ``Configuration``, named ``child``, pointing to ``NestedChild``.
Likewise, ``NestedChild`` has a child ``Configuration``, named ``child``.

.. note::
    Children hyper-parameters are identified by the ``Parameter`` field ``is_child``.
    If this field is not set to ``True``, cinnamon has no way to understand whether that ``Parameter`` is pointing to a
    ``Configuration`` or not.

.. note::
    **Configuration variants** also supports nesting! In particular, the ``add_and_bind_variants`` also checks for children ``Configuration`` and their specified variants.