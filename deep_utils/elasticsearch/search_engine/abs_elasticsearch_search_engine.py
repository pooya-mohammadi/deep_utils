from enum import Enum
from typing import Union


class ElasticSearchABS:
    class QueryKeyword(str, Enum):
        MATCH = "match"
        MATCH_ALL = "match_all"
        MATCH_PHRASE = "match_phrase"

    @staticmethod
    def get_match_query(field_name: str = "",
                        field_value: str = "",
                        keyword: Union[QueryKeyword, str] = "match",
                        ):
        """
        This function is used to get query-match. It will simply return query-match json
        :param field_name:
        :param field_value:
        :param keyword:
        :return:
        """
        if keyword == ElasticSearchABS.QueryKeyword.MATCH:

            query = {
                keyword: {
                    field_name: field_value
                }
            }
        elif keyword == ElasticSearchABS.QueryKeyword.MATCH_ALL:
            query = {
                keyword: {}
            }
        elif keyword == ElasticSearchABS.QueryKeyword.MATCH_PHRASE:
            query = {
                keyword: {
                    field_name: field_value
                }
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
