.. _registration:

Registration APIs
*********************************************

Let's first recap the core concepts of ``cinnamon``.

**Configuration**
    defines all hyper-parameters and conditions of a Component

**Component**
    defines a code logic

Cinnamon allows to quickly **retrieve**, **build** and **re-use** a ``Component`` via a set of registration APIs.

Consider our data loader recurring example.

.. code-block:: python

    # Component
    class DataLoader(Component):

        def load(...):
            ...

    # Configuration
    class DataLoaderConfig(Configuration):

        @classmethod
        def get_default(cls):
            config = super().get_default(cls)
            config.add(name='folder_name',
                       type_hint=str,
                       is_required=True)

            return config

*********************************************
Registering a Configuration
*********************************************

Suppose we have a <``Configuration``, ``Component``> pair (e.g., <``DataLoaderConfig``, ``DataLoader``>).

We first store the ``Configuration`` in the cinnamon's ``Registry`` to memorize it.

We do so by defining a ``RegistrationKey`` to uniquely access to the ``Configuration``.

.. image:: /img/registration_with_key.png
    :scale: 60%
    :align: center


We register the ``DataLoaderConfig`` as follows:

.. code-block:: python

    Registry.add_configuration(config_class=DataLoaderConfig,
                               name='data_loader'
                               namespace='showcase')

Or, we can define the ``RegistrationKey`` explicitly

.. code-block:: python

    key = RegistrationKey(name='data_loader', namespace='showcase')
    Registry.add_configuration_from_key(config_class=DataLoaderConfig,
                                        key=key)


Once registered, we can always **retrieve and build** a ``Configuration`` from the ``Registry``.

.. code-block:: python

    config = Registry.build_configuration(name='data_loader',
                                          namespace='showcase')

or

.. code-block:: python

    key = RegistrationKey(name='data_loader', namespace='showcase')
    config = Registry.build_configuration_from_key(key=key)

In particular, the ``get_default()`` method is invoked to build the ``Configuration`` instance.

*********************************************
Custom Configuration instance building
*********************************************

The ``Configuration.get_default()`` defines the general structure of a ``Configuration`` (see :ref:`configuration` for more details).

What if we want to build a ``Configuration`` instance via a **custom function**?

Suppose the following ``DataLoaderConfig``:

.. code-block:: python

    # Configuration
    class DataLoaderConfig(Configuration):

        @classmethod
        def get_default(cls):
            config = super().get_default(cls)
            config.add(name='folder_name',
                       type_hint=str,
                       is_required=True)

            return config

       @classmethod
       def limited_samples_variant(cls):
            config = cls.get_default()
            config.folder_name = '*folder_name"'
            config.add(name='max_samples_amount', value=100, type_hint=int)
            return config


We want to register our ``Configuration`` such that its instances are built via the ``limited_samples_variant()`` method.

We do so by specifying the ``DataLoaderConfig.limited_samples_variant`` constructor method when registering the ``DataLoaderConfig``.

.. code-block:: python

    Registry.add_configuration(name='data_loader',
                               namespace='showcasing',
                               config_constructor=DataLoaderConfig.limited_samples_variant)


*********************************************
Registration Key
*********************************************

A ``RegistrationKey`` is a unique compound identifier that allows to quickly retrieve a ``Configuration`` from the
``Registry``.

In particular, a ``RegistrationKey`` consists of

*   **name**: a generic name to identify the type of configuration (and corresponding bound component, if any). For instance, 'data_loader' for a data loader.

*   **[Optional] tags**: a set of string tags to identify the configuration. For instance, two data loaders will have the same name but different tags.

*   **namespace**: the namespace to which the configuration belongs to. For instance, two configurations pointing to the same deep learning model, one written in Tensorflow and the other one in Pytorch, have the same name and tags but different namespace.


*********************************************
Binding a Configuration to a Component
*********************************************

Once we have registered our ``Configuration``, we need to **bind** it to a ``Component`` to automatically build
``Component`` instances.

We instruct the ``Registry`` to perform the binding operation by leveraging the ``RegistrationKey`` used to
store our ``Configuration``.

.. image:: img/binding_with_key.png
    :scale: 60%
    :align: center

In our data loader example, we perform the binding between the registered ``DataLoaderConfig`` and ``DataLoader`` as follows

.. code-block:: python

    Registry.bind(component_class=DataLoader,
                  name='data_loader',
                  namespace='showcase')

or

.. code-block:: python

    key = RegistrationKey(name='data_loader', namespace='showcase')
    Registry.bind_from_key(component_class=DataLoader,
                           key=key)


The ``Registry`` offers the capability of performing the **registration** and **binding** operations in one step.

.. code-block:: python

    Registry.add_and_bind(config_class=DataLoaderConfig,
                          component_class=DataLoader,
                          name='data_loader',
                          namespace='showcase')

If the ``DataLoaderConfig`` has some hyper-parameter variants to take into account, we can register them as well

.. code-block:: python

    Registry.add_and_bind_variants(config_class=DataLoaderConfig,
                                   component_class=DataLoader,
                                   name='data_loader',
                                   namespace='showcase')


*********************************************
Building a Component
*********************************************

Once a ``Configuration`` is bound to a ``Component``, the ``Registry`` can automatically build a ``Component`` instance
by using the associated ``RegistrationKey``.

.. code-block:: python

    data_loader = Registry.build_component(name='data_loader',
                                           namespace='showcasing')

or

.. code-block:: python

    key = RegistrationKey(name='data_loader', namespace='showcasing')
    data_loader = Registry.build_component_from_key(registration_key=key)

.. note::
    The ``Registry`` deals with ``Configuration`` and ``Component`` **classes** and **not** class instances.
    Classes are stored as ''factories'' for building class instances on-the-fly.

*********************************************
Retrieving a Component
*********************************************

The ``Registry`` can also retrieve the ``Component`` class instead of building an instance

.. code-block:: python

    component_class = Registry.retrieve_component(name='data_loader',
                                                  namespace='showcasing')

or

.. code-block:: python

    key = RegistrationKey(name='data_loader', namespace='showcasing')
    component_class = Registry.retrieve_component_from_key(registration_key=key)


**************************************************
Registering and retrieving a Component instance
**************************************************

A ``Component`` instance can be registered as well via a ``RegistrationKey``.

.. note::
    The same ``RegistrationKey`` used to bind the ``Component`` can be used as well.

Such a functionality is particularly useful to have access to a ``Component`` instance anywhere in the code.

.. code-block:: python

    Registry.register_component_instance(name='data_loader',
                                         namespace='showcasing',
                                         component=component)           # instantiated somewhere


or

.. code-block:: python

    key = RegistrationKey(name='data_loader', namespace='showcasing')
    Registry.register_component_instance_from_key(registration_key=key
                                                  component=component)



Additionally, we can directly register the ``Component`` instance when building it.

.. code-block:: python

    component = Registry.build_component(name='data_loader',
                                         namespace='showcasing',
                                         register_component_instance=True)


Once registered, we can always retrieve the ``Component`` instance via the associated ``RegistrationKey``

.. code-block:: python

    component = Registry.retrieve_component_instance(name='data_loader',
                                                     namespace='showcasing')

or

.. code-block:: python

    key = RegistrationKey(name='data_loader', namespace='showcasing')
    component = Registry.retrieve_component_instance_from_key(registration_key=key)

*********************************************
Empty Configuration
*********************************************

In some cases, a ``Component`` may not have any hyper-parameters.

We can use the ``Configuration`` class to bind an empty ``Configuration``.

.. code-block:: python

    Registry.add_and_bind(config_class=Configuration,
                          component_class=DataLoader,
                          name='data_loader',
                          namespace='showcasing')


*********************************************
Tl;dr (Too long; didn't read)
*********************************************

- Define your ``Component`` (code logic).
- Define its corresponding ``Configuration`` (one or more).
- Register the ``Configuration`` to the ``Registry`` via a ``RegistrationKey``.
- The ``RegistrationKey`` is a compound string-based unique identifier.
- Bind the ``Configuration`` to its ``Component`` via the ``RegistrationKey``.
- Build ``Component`` instances via the ``RegistrationKey``.

**Congrats! This is 99% of cinnamon!**

*********************************************
How to use registration APIs
*********************************************

You may be wondering **how** to properly use these registration APIs...

Long story short, you **don't need** to contaminate your code with registration and binding operations.

Cinnamon supports a **specific code organization** to **automatically** address all registration related operations while keeping a clean code organization.

See :doc:`dependencies` for more details.