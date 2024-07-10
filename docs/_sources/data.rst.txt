.. _data:

Data
*********************************************

Cinnamon offers utility APIs for enforcing type checks, conditions and other control flows to other thing rather than ``Component`` and ``Configuration``.

In particular, the ``FieldDict`` class is like a ``Configuration`` without ``Parameter``.

.. code-block:: python

    field_dict = FieldDict()
    field_dict.add(name='x',
                   value=42,
                   type_hint=int,
                   description="An unknown variable")

The ``FieldDict`` can be used as a return type of ``Component`` or other code snippets to enforce code correctness.

Fields in a ``FieldDict`` can be referenced as ``Parameter`` are referenced in ``Configuration

.. code-block:: python

    print(field_dict.x)                 # 42
    field_dict.x = 50
    print(field_dict.x)                 # 50
    print(field_dict['x'])              # 50
    print(type(field_dict.get('x')))    # Field

*********************************************
Adding Conditions
*********************************************

Just like a ``Configuration``, we can add conditions to ``FieldDict``

.. code-block:: python

    field_dict = FieldDict()
    field_dict.add(name='x',
                   value=[1, 2, 3],
                   type_hint=List[int],
                   description="An unknown variable")
    field_dict.add(name='y',
                   value=[2, 2, 2],
                   type_hint=List[int],
                   description="Another unknown variable")
    field_dict.add_condition(condition=lambda fields: len(fields.x) == len(fields.y),
                             name='x_y_pairing')

The above condition enforces that the lengths of ``x`` and ``y`` must be equal.

We can validate the above condition, by **validating** the ``FieldDict``

.. code-block:: python

    field_dict.validate()