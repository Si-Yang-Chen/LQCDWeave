"""Product names and endpoint axes shared by PR numerical and IO services."""

ALL_PRODUCTS = ('VSV', 'VSP', 'VSD', 'PSV', 'PSP', 'PSD', 'DSV', 'DSP', 'DSD')
SAVED_PRODUCTS = ('VSV', 'PSV', 'PSP', 'VSD', 'PSD', 'DSD')


def normalize_product(product):
    product = str(product).upper()
    if product not in ALL_PRODUCTS:
        raise ValueError('product must be one of ' + ', '.join(ALL_PRODUCTS))
    return product


def reverse_product(product):
    product = normalize_product(product)
    return product[2] + 'S' + product[0]


def endpoint_axes(kind, side):
    if kind == 'P':
        return (side + '_point', side + '_color')
    return (side + ('_mode' if kind == 'V' else '_dilution_vector'),)


def product_axes(product):
    """Single physical-pair axes, preserving complete point/color groups."""
    product = normalize_product(product)
    return (('sink_spin', 'source_spin') + endpoint_axes(product[0], 'sink')
            + endpoint_axes(product[2], 'source'))


def product_shape(product, sink_count, source_count):
    product = normalize_product(product)
    sink = (sink_count, 3) if product[0] == 'P' else (sink_count,)
    source = (source_count, 3) if product[2] == 'P' else (source_count,)
    return (4, 4) + sink + source
