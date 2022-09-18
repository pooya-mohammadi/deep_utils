import os
from typing import Dict, Union
from elasticsearch import Elasticsearch
from deep_utils.utils.logging_utils.logging_utils import value_error_log
from deep_utils.utils.json_utils.json_utils import dump_json


class ElasticsearchEngin:
    def __init__(self, elastic_url="http://localhost:9200", es=None, logger=None, verbose=1):
        if es is None:
            self.es = Elasticsearch(elastic_url)
        elif isinstance(es, Elasticsearch):
            self.es = es
        else:
            value_error_log(logger, "url or es instance are not provided")
        self.verbose = verbose

    def get_suggestion_output(self, text, field_name, index_name="tehran-roads-centers"):
        query = {
            "match": {
                field_name: text
            }
        }
        suggest = {
            "my-suggestion": {
                "field_value": text,
                "term": {
                    "field": field_name
                }
            }
        }
        results = self.es.search(index=index_name, query=query, suggest=suggest)
        hits = results['hits']['hits']
        if len(hits) > 1:
            return text
        options = results['suggest']['my-suggestion'][0]['options']
        if len(options) == 0:
            # return the exact field_value
            return text
        else:
            return options[0]['field_value']

    # def full_search_w_parents(self, city, index_name, province_id, search_field, area_type_start=None):
    #     _id, area_type = None, None
    #     if area_type_start is None:
    #         for area_type in self.down_to_top:
    #             result = self.term_search(self.es, {search_field: city,
    #                                                 "province_id": province_id,
    #                                                 "area_type": area_type},
    #                                       index_name=index_name)
    #             _id, area_type = self._get_id(result, area_type=True)
    #             if _id:
    #                 break
    #     else:
    #         area_type = area_type_start
    #         while True:
    #             result = self.term_search(self.es, {search_field: city,
    #                                                 "province_id": province_id,
    #                                                 "area_type": area_type},
    #                                       index_name=index_name)
    #             _id, area_type = self._get_id(result, area_type=True)
    #             area_type = self.area_type_parents[area_type]
    #             if _id or area_type is None:
    #                 break
    #     return _id, area_type

    @staticmethod
    def get_index_value_hits(hits, index, field_name):
        if not hits:
            return None
        else:
            return hits[index]['_source'].get(field_name, None)

    @staticmethod
    def _get_id(result: dict, area_type=False):
        hits = result["hits"]['hits']
        if len(hits) == 0:
            if area_type:
                return None, None
            return None
        if len(hits) > 1:
            value = hits[0]["_source"]
            province = value['province_name']
            city_name = value['name']
            os.makedirs("equals", exist_ok=True)
            dump_json(f"equals/{province}_{city_name}.json", hits, ensure_ascii=False)
            print(f"There are two objects with the same characteristics with the same name, {hits}")
            hits = [hits[0]]

        if len(hits) == 1:
            value = hits[0]["_source"]
            _id = value['id']
            if area_type:
                area_type = value['area_type']
                return _id, area_type
            else:
                return _id
        else:
            raise ValueError("We've got a problem")

    @staticmethod
    def term_search(es_client, field_term_dict: dict, index_name: str):

        query = ElasticsearchEngin.get_bool_must_constant_score_query(field_term_dict)

        results = es_client.search(index=index_name, query=query)
        return results.body

    # def search_query_sort(self, field_term_dict, lat=None, lon=None, index_name:str):
    #     query = self.get_bool_must_constant_score_query(field_term_dict)
    #     if lat and lon:
    #         geo_sort = self.get_geo_sort(lat, lon)
    #         sort = [geo_sort]
    #     else:
    #         fclass_lev_sort = self.get_sort("fclass_lev")
    #         sort = [fclass_lev_sort]
    #     results = self.es.search(index=index_name, query=query, sort=sort).body
    #     hits = results['hits']['hits']
    #     if len(hits) == 0:
    #         return None
    #     source = hits[0]['_source']
    #     location = source['location']
    #     full_address = source['address']
    #     return location["lat"], location['lon'], full_address

    @staticmethod
    def get_geo_sort(x, y, field_name="location", order="asc"):
        if x and y:
            geo_sort = {
                "_geo_distance": {
                    field_name: [x, y],
                    "order": order,
                    "unit": "km"
                }
            }
            sort = [geo_sort]
        else:
            sort = []

        return sort

    @staticmethod
    def get_sort_query(field_name, order="asc"):
        """
        This function creates a sort request
        :param field_name: The field to sort
        :param order: the order. Default is asc.
        :return:
        """
        if order and field_name:
            sort = {
                field_name: {
                    "order": order,
                }
            }
            sort = [sort]
        else:
            sort = []
        return sort

    @staticmethod
    def get_bool_must_constant_score_query(field_term_dict: dict):
        must_list = []
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
    def get_bool_must_fuzzy_query(field_term_dict: Dict[str, str]):
        must_list = []
        for field, term in field_term_dict.items():
            if term is None:
                continue
            inner_query = {
                "fuzzy": {
                    field: {
                        "value": term
                    }
                }
            }
            must_list.append(inner_query)

        query = {"bool": {"must": must_list}}
        return query

    def search_bool_must_fuzzy_query(self,
                                     field_term_dict: dict,
                                     index_name: str,
                                     size: Union[int, None] = None):
        query = self.get_bool_must_fuzzy_query(field_term_dict)
        results = self.es.search(index=index_name, query=query, size=size)
        hits = ElasticsearchEngin.get_hits(results)
        return hits

    @staticmethod
    def get_hits(results):
        """
        This is a simple method to extract hits or return none when the output of the search has no hits
        :param results:
        :return:
        """
        hits = results['hits']['hits']
        if len(hits) == 0:
            hits = None
        return hits

    def search_bool_must_fuzzy_query_geo_sort(self,
                                              field_term_dict: dict,
                                              index_name: str,
                                              lat: float,
                                              lon: float,
                                              geo_sort_field_name="location",
                                              geo_sort_order="asc",
                                              size: Union[int, None] = None):
        """
        :param field_term_dict:
        :param index_name:
        :param lat:
        :param lon:
        :param geo_sort_field_name:
        :param geo_sort_order:
        :param size:
        :return:
        """
        assert lat is not None and lon is not None
        query = self.get_bool_must_fuzzy_query(field_term_dict)
        sort = self.get_geo_sort(lat, lon, field_name=geo_sort_field_name, order=geo_sort_order)
        results = self.es.search(index=index_name, query=query, sort=sort, size=size).body
        hits = self.get_hits(results)
        return hits

    def get_bool_must_constant_score_query_geo_sort(self,
                                                    field_term_dict: dict,
                                                    index_name: str,
                                                    lat: float, lon: float,
                                                    size: Union[int, None] = None):
        assert lat is not None and lon is not None
        query = self.get_bool_must_constant_score_query(field_term_dict)
        sort = self.get_geo_sort(lat, lon)
        results = self.es.search(index=index_name, query=query, sort=sort, size=size).body
        hits = self.get_hits(results)
        return hits

    def search_bool_must_fuzzy_query_sort(self,
                                          field_term_dict: dict,
                                          index_name: str,
                                          sort_filed: str,
                                          sort_order: str = "asc",
                                          size: Union[int, None] = None):
        """
        This method searches based on fuzzy based
        GET iran-roads/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "fuzzy": {
            "file-name": {
              "value": "field-value"
            }
          }
        },
        {
          "fuzzy": {
            "file-name-2": {
              "value": "field-value-2"
            }
          }
        }
      ]
    }
  },
  "sort": [
    {
      "sort-field-name": {
        "order": "desc"
      }
    }
  ]
}
        :param field_term_dict:
        :param index_name:
        :param sort_filed:
        :param sort_order:
        :param size:
        :return:
        """
        assert sort_filed is not None and sort_order is not None
        query = self.get_bool_must_fuzzy_query(field_term_dict)
        sort = self.get_sort_query(field_name=sort_filed, order=sort_order)
        results = self.es.search(index=index_name, query=query, sort=sort, size=size).body
        hits = self.get_hits(results)
        return hits

    def search_bool_must_fuzzy_constant_score_range_sort(self,
                                                         field_term_dict: dict,
                                                         index_name: str,
                                                         sort_filed: str,
                                                         sort_order: str = "asc",
                                                         size: Union[int, None] = None):
        assert sort_filed is not None and sort_order is not None
        query = self.get_bool_must_constant_score_query(field_term_dict)
        sort = self.get_sort_query(field_name=sort_filed, order=sort_order)
        results = self.es.search(index=index_name, query=query, sort=sort, size=size).body
        hits = ElasticsearchEngin.get_hits(results)
        return hits

    @staticmethod
    def get_match_query(field_name,
                        field_value,
                        ):
        """
        This function is used to get query-match. It will simply return query-match json
        :param field_name:
        :param field_value:
        :return:
        """
        query = {
            "match": {
                field_name: field_value
            }
        }
        return query

    def search_query_match_sort(self, index_name, field_name, field_value, sort_field_name, sort_order, size=None):
        """
        This method is used to do a simple query match with sort.

        :param index_name:
        :param field_name:
        :param field_value:
        :param sort_field_name:
        :param sort_order:
        :param size:
        :return:
        """
        query_match = ElasticsearchEngin.get_match_query(field_name, field_value)
        sort = ElasticsearchEngin.get_sort_query(field_name=sort_field_name, order=sort_order)
        results = self.es.search(index=index_name, query=query_match, sort=sort, size=size).body
        hits = ElasticsearchEngin.get_hits(results)
        return hits

    def search_query_match_geo_sort(self, index_name, field_name, field_value, geo_sort_field_name, geo_sort_order, lat,
                                    lon,
                                    size=None):
        """
        This method is used to do a simple query match with geo_sort.

        :param index_name:
        :param field_name:
        :param field_value:
        :param geo_sort_field_name:
        :param geo_sort_order:
        :param lat:
        :param lon:
        :param size:
        :return:
        """
        query_match = ElasticsearchEngin.get_match_query(field_name, field_value)
        geo_sort = ElasticsearchEngin.get_geo_sort(lat, lon, geo_sort_field_name, geo_sort_order)
        results = self.es.search(index=index_name, query=query_match, sort=geo_sort, size=size).body
        hits = ElasticsearchEngin.get_hits(results)
        return hits

    def search_query_match_sort_geo_sort(self, index_name, field_name, field_value, geo_sort_field_name, geo_sort_order,
                                         sort_field_name, sort_order, lat, lon, size=None):
        """
        This method is used to do a simple query match with geo_sort plus a usual sort
        :param index_name:
        :param field_name:
        :param field_value:
        :param geo_sort_field_name:
        :param geo_sort_order:
        :param sort_field_name:
        :param sort_order:
        :param lat:
        :param lon:
        :param size:
        :return:
        """
        query_match = ElasticsearchEngin.get_match_query(field_name, field_value)
        geo_sort = ElasticsearchEngin.get_geo_sort(lat, lon, geo_sort_field_name, geo_sort_order)
        sort = ElasticsearchEngin.get_sort_query(field_name=sort_field_name, order=sort_order)
        results = self.es.search(index=index_name, query=query_match, sort=geo_sort + sort, size=size).body
        hits = ElasticsearchEngin.get_hits(results)
        return hits
