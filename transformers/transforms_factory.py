EMPTY_NAME_ERR = 'Name of transformer or one of its arguments cant be empty\n\
                  Use "name/arg1=value/arg2=value" format'
                  
def parse_transformers(transformers):
    """
    Parse the list of transformers, given by configuration, into a list of
    tuple of the transformers name and a dictionary containing additional args.

    The transformer is assumed to be of the form 'name/arg1=value/arg2=value'

    :raw_transformers: list of strings [unparsed transformers]
    :returns: list of parsed transformers [list of (name,additional_args)]
    """
    raw_transformers = transformers

    transformers = []
    for t in raw_transformers:
        arguments = t.split('/')
        name = arguments[0]
        if name == '':
            raise Exception(EMPTY_NAME_ERR)

        kwargs = {}
        if len(arguments) > 1:
            for a in arguments[1:]:
                splited = a.split('=')
                var = splited[0]
                val = splited[1] if len(splited) > 1 else None
                if var == '':
                    raise Exception(EMPTY_NAME_ERR)

                kwargs[var] = float(val)

        transformers.append((name, kwargs))

    return transformers