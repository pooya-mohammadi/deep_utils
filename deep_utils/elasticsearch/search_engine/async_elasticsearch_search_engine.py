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

    @staticmethod
    def get_match_query(field_name="",
                        field_value="",
                        keyword="match",
                        ):
        """
        This function is used to get query-match. It will simply return query-match json
        :param field_name:
        :param field_value:
        :param keyword:
        :return:
        """
        if keyword == "match":

            query = {
                keyword: {
                    field_name: field_value
                }
            }
        elif keyword == "match_all":
            query = {
                keyword: {}
            }
        else:
            raise ValueError(f"keyword: {keyword} is not valid!")
        return query

    @staticmethod
    def get_hits(results, return_source: bool = False):
        """
        This is a simple method to extract hits or return none when the output of the search has no hits
        :param results:
        :param return_source: If set to true the _source keyword will be returned!
        :return:
        """
        hits = results['hits']['hits']
        if len(hits) == 0:
            hits = None
        elif return_source:
            hits = [hit["_source"] for hit in hits]
        return hits

    async def search_match_query(self, index_name, field_name="", field_value="", keyword="match", size=None,
                                 return_source: bool = False):
        """
        A simple match search
        {
        "query": {
            "keyword": {
                "field_name": "field_value"
                     }
                 }
         }
        :param index_name:
        :param field_name:
        :param field_value:
        :param keyword:
        :param size:
        :param return_source:
        :return:

        """
        query = self.get_match_query(field_name=field_name, field_value=field_value, keyword=keyword)
        results = await self.es.search(index=index_name, query=query, size=size)
        hits = AsyncElasticsearchEngin.get_hits(results.body, return_source=return_source)
        return hits
