from enum import Enum
from typing import Union, Any


class ElasticSearchABS:
    class QueryKeyword(str, Enum):
        MATCH = "match"
        MATCH_ALL = "match_all"
        MATCH_PHRASE = "match_phrase"

    @staticmethod
    def get_bool_must_constant_score_query(field_term_dict: dict[str, int | float | str]):
        """
        creates query for must list or query list!
        :param field_term_dict:
        :return:
        """
        must_list = []
        field_term_dict = field_term_dict or dict()
        for field, term in field_term_dict.items():
            if term is None:
                continue
            inner_query = {
                "constant_score": {
                    "filter": {
                        "term": {
                            field: term
                        }
                    }
                }
            }
            must_list.append(inner_query)

        query = {"bool": {"must": must_list}}
        return query

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
        keyword = keyword.value if isinstance(keyword, ElasticSearchABS.QueryKeyword) else keyword
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

    @staticmethod
    def get_match_and_fileter_query(field_name: str,
                                    field_value: str,
                                    filter_name: str,
                                    filter_value: str,
                                    match_keyword: QueryKeyword = QueryKeyword.MATCH
                                    ) -> dict[str, Any]:
        """
        This function is used to get query for match and must together
        {
          "query": {
            "bool": {
              "filter": {
                "term": {
                  "field1": "value1"
                }
              },
              "must": {
                "match": {
                  "field2": "value2"
                }
              }
            }
          }
        }
        :return:
        """
        match_keyword = match_keyword.value if isinstance(match_keyword,
                                                          ElasticSearchABS.QueryKeyword) else match_keyword
        query = {
            "bool": {
                "should": [
                    {
                        match_keyword: {
                            field_name: field_value
                        }
                    }
                ],
                "filter": {
                    "term": {filter_name: filter_value}
                }
            }
        }
        return query
