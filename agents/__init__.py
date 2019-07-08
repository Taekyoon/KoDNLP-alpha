class Agent(object):
    def _run(self, query: str):
        raise NotImplementedError()

    def __call__(self, query: str):
        return self._run(query)
