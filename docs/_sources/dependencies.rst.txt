.. _dependencies:

Code Organization
*********************************************

We recommend organizing your code as follows

.. code-block::

    project_folder
        configurations
            folder containing ``Configuration`` scripts

        components
            folder containing ``Component`` scripts

We also recommend using the same filename for <``Configuration``, ``Component``> paired scripts for readability purposes.

Recalling our data loader recurring example, we have

.. code-block::

    project_folder
        configurations
            data_loader.py

        components
            data_loader.py

where

configurations/data_loader.py
    .. code-block:: python

        class DataLoader(Component):

            def load(...):
                ...

components/data_loader.py
    .. code-block:: python

        class DataLoaderConfig(Configuration):

            @classmethod
            def get_default(cls):
                config = super().get_default(cls)
                config.add(name='folder_name',
                           type_hint=str,
                           is_required=True)

                return config



*********************************************
Registration calls
*********************************************

Cinnamon **requires** registration APIs to be **located** in configuration script files, **wrapped** into python functions, and **decorated** with ``@register`` decorator.

For instance, ``configurations/data_loader.py`` should be as follows:

.. code-block:: python

    class DataLoaderConfig(Configuration):

        @classmethod
        def get_default(cls):
            config = super().get_default(cls)
            config.add(name='folder_name',
                       type_hint=str,
                       is_required=True)

            return config

    @register
    def register_data_loaders():
        Registry.add_and_bind(config_class=DataLoaderConfig,
                              component_class=DataLoader,
                              name='data_loader',
                              namespace='showcase')

*********************************************
Dependency DAG
*********************************************

This code organization is meant to simplify registration burden while keeping high readability.

The ``Registry`` can be issued to look for all ``@register`` decorated functions located in ``configurations`` folder
to automatically execute them.

.. code-block:: python

    Registry.check_registration_graph()
    Registry.expand_and_resolve_registration()


The first function checks if the registration DAG is valid. Indeed, registration APIs like ``add_and_bind`` or ``add_configuration`` issue a **delayed registration action** to avoid conflicts.

This means that the ``Registry`` first **builds a graph** where nodes are ``RegistrationKey`` and links denote a dependency. Then the ``Registry`` **checks** if the graph is a DAG (i.e., it has no loops)

The ``Registry`` eventually issues all registration function calls in order according to the dependency graph (``expand_and_resolve_registration()``)

The dependency DAG is necessary since the ``Registry`` doesn't know the **correct registration order**.
Additionally, as the number of registrations increases, it becomes cumbersome to keep track of all possible valid registration orders.

**Cinnamon does that for you!**

One can inspect the generated dependency DAG as follows

.. code-block:: python

    Registry.show_dependencies()

This method generates a ``dependencies.html`` containing a graphical representation of the dependency DAG, useful for debugging.


*********************************************
External registrations
*********************************************

Cinnamon is a community project. This means that **you** are the main contributor.

In many situations, you may need to import other's work: external configurations and components.

Cinnamon supports loading registration function calls that are external to your project's ``configurations`` folder.
Moreover, you can also build your ``Configuration`` and ``Component`` with dependencies on external ones.

For instance, suppose that a ``DataLoaderConfig`` variant has a external child (i.e., a ``Parameter`` pointing to an external ``RegistrationKey``).

.. code-block:: python

    class DataLoaderConfig(Configuration):

        @classmethod
        def get_default(cls):
            config = super().get_default(cls)
            config.add(name='folder_name',
                       type_hint=str,
                       is_required=True)

            return config

        @classmethod
        def external_variant(cls):
            config = cls.get_default()
            config.add(name='processor',
                       namespace='external')

    @register
    def register_data_loaders():
        Registry.add_and_bind(config_class=DataLoaderConfig,
                              component_class=DataLoader,
                              config_constructor=DataLoaderConfig.external_variant,
                              name='data_loader',
                              namespace='showcase')


This registration is possible if we tell the ``Registry`` where to retrieve the ``RegistrationKey`` with ``name='processor'`` and ``namespace='external'``
We can do so via ``Registry.load_registrations()`` to be invoked at the **beginning** of our main script to execute.


.. code-block:: python

    external_directory_path = ...
    Registry.load_registrations(directory_path=external_directory_path)

In this way, during the dependency DAG resolution and expansion, the ``Registry`` searches in ``external_directory_path`` folder for ``RegistrationKey`` that are not found locally (i.e., in ``configurations`` folder).