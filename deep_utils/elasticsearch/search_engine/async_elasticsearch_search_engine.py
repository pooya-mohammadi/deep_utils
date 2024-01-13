from elasticsearch import AsyncElasticsearch

from deep_utils.utils.logging_utils.logging_utils import value_error_log


class AsyncElasticsearchEngin:
    def __init__(self, elastic_url="http://localhost:9200", es: AsyncElasticsearch = None, timeout=10, logger=None,
                 verbose=1):
        if es is None:
            self.es = AsyncElasticsearch(elastic_url, timeout=timeout)
        elif isinstance(es, AsyncElasticsearch):
            self.es = es
        else:
            value_error_log(logger, "url or es instance are not provided")
        self.verbose = verbose

    async def match_all(self, index: str, size: int = 20):
        """
        query={"match_all": {}}, size=size
        :param index:
        :param size:
        :return:
        """
        resp = await self.es.search(
            index=index,
            query={"match_all": {}},
            size=size,
        )
        return resp

