from cinnamon.configuration import Configuration
from cinnamon.registry import register, Registry

from mamkit.components.collators import LabelCollator


@register
def register_collators():
    Registry.register_configuration(name='collator',
                                    namespace='mamkit',
                                    tags={'label'},
                                    config_class=Configuration,
                                    component_class=LabelCollator)
