from cinnamon.registry import RegistrationKey


def match_compound_tags(a_key, b_key):
    if a_key is None or b_key is None:
        return False

    if not isinstance(a_key, RegistrationKey) or not isinstance(b_key, RegistrationKey):
        return True

    return not a_key.compound_tags.difference(b_key.compound_tags)
