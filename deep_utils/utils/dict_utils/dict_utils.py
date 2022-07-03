class frozendict:
    def __init__(self, **kwargs):
        super(frozendict, self).__init__()
        self.repo = dict()
        for key, item in kwargs.items():
            self.repo[key] = item

    def __getitem__(self, item):
        return self.repo[item]

    def __setitem__(self, key, value):
        raise TypeError(f"frozendict object does not support updating")