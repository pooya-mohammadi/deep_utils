from re import sub


class PyUtils:

    @staticmethod
    def camel2snake_case(camel_case):
        """
        Convert camel case to snake case
        :param camel_case:
        :return:
        >>> PyUtils.camel2snake_case('CamelCase')
        'camel_case'
        >>> PyUtils.camel2snake_case('ConfigClass')
        'config_class'
        """
        return '_'.join(
            sub('([A-Z][a-z]+)', r' \1',
                sub('([A-Z]+)', r' \1',
                    camel_case.replace('-', ' '))).split()).lower()

    @staticmethod
    def static_upper_case2snake_case(static_upper_case: str):
        """
        Convert static upper case to snake case
        :param static_upper_case:
        :return:
        >>> PyUtils.static_upper_case2snake_case('STATIC_UPPER_CASE')
        'static_upper_case'
        """
        return static_upper_case.lower()
