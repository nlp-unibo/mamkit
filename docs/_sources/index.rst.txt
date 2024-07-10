.. cinnamon-core documentation master file, created by
   sphinx-quickstart on Sun May 21 18:00:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cinnamon Core Package
================================================

Consider a code logic that has to load some data.

.. code-block:: python

   class DataLoader:

      def load(...):
          data = read_from_file(folder_name=self.folder_name)
          return data

The data loader reads from a file located according to ``self.folder_name``'s value.

If ``self.folder_name`` has multiple values, we can use the same code logic to load data from different folders.

Hypothetically, we would define multiple data loaders:

.. code-block:: python

   data_loader_1 = DataLoader(folder_name='*folder_name1*')
   data_loader_2 = DataLoader(folder_name='*folder_name2*')
   ...

Now, if the data loader code block is used in a project, we require some code modularity to avoid
defining several versions of the same script.
One common solution is to rely on **configuration files** (e.g., JSON file).

.. code-block:: python

   {
      'data_loader' : {
         'folder_name': '*folder_name1*'
      }
   }

The main script is modified to load our configuration file so that each code logic is properly initialized.

===============================================
Cinnamon
===============================================

Well, cinnamon keeps this <configuration, code logic> dichotomy where a configuration is written in **plain Python code**!

.. code-block:: python

   class DataLoaderConfig(Configuration):

      @classmethod
      def get_default(cls):
         config = super().get_default()

         config.add(name='folder_name',
                    type_hint=str,
                    is_required=True,
                    variants=['*folder_name1*', '*folder_name2*', ...],
                    description="Base folder name from which to look for data files.")
         return config


Cinnamon allows **high-level configuration definition** (constraints, type-checking, description, variants, etc...)

To quickly load any instance of our data loader code logic, we

**register**
   the configuration via a **registration key**

   .. code-block:: python

      key = RegistrationKey(name='data_loader', namespace='showcase')
      Registry.add_configuration_variants_from_key(config_class=DataLoaderConfig,
                                 key=key)

**bind**
   the configuration to its code logic: ``DataLoaderConfig`` --> ``DataLoader``

   .. code-block:: python

      Registry.bind(config_class=DataLoaderConfig,
                    component_class=DataLoader,
                    key=key)

**build**
   the ``DataLoader`` code logic with a specific configuration instance via the used **registration key**

   .. code-block:: python

      variant_key = RegistrationKey(name='data_loader', tags=['folder_name=*folder_name1*'], namespace='showcase')
      data_loader = Registry.build_component(key=variant_key)


**That's it!** This is all of you need to use cinnamon.


Install
===============================================

pip
   .. code-block:: bash

      pip install cinnamon-core

git
   .. code-block:: bash

      git clone https://github.com/federicoruggeri/cinnamon_core


===============================================
Contribute
===============================================

Want to contribute with new ``Component`` and ``Configuration``?

Feel free to submit a merge request!

Cinnamon is meant to be a community project :)

===============================================
Contact
===============================================

Don't hesitate to contact:
- `Federico Ruggeri <federico.ruggeri6@unibo.it>`_

for questions/doubts/issues!

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Contents:
   :titlesonly:

   Registration <registration.rst>
   Code Organization <dependencies.rst>
   Configuration <configuration.rst>
   Component <component.rst>
   Data <data.rst>
   cinnamon-core <cinnamon_core.rst>