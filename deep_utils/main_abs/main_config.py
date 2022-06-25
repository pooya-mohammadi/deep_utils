class MainConfig:
    def vars(self) -> dict:
        out = dict()
        for key in dir(self):
            val = getattr(self, key)
            if (key.startswith("__") and key.endswith("__")) or type(
                val
            ).__name__ == "method":
                continue
            else:
                out[key] = val
        return out

    def __repr__(self):
        configs = self.vars()
        view = f"{self.__class__.__name__} -> " + ", ".join(
            f"{k}: {v}" for k, v in configs.items()
        )
        return view
