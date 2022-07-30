import os
import string
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
                "text": text,
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
            # return the exact text
            return text
        else:
            return options[0]['text']

    def full_search_w_parents(self, city, index_name, province_id, search_field, area_type_start=None):
        _id, area_type = None, None
        if area_type_start is None:
            for area_type in self.down_to_top:
                result = self.term_search(self.es, {search_field: city,
                                                    "province_id": province_id,
                                                    "area_type": area_type},
                                          index_name=index_name)
                _id, area_type = self._get_id(result, area_type=True)
                if _id:
                    break
        else:
            area_type = area_type_start
            while True:
                result = self.term_search(self.es, {search_field: city,
                                                    "province_id": province_id,
                                                    "area_type": area_type},
                                          index_name=index_name)
                _id, area_type = self._get_id(result, area_type=True)
                area_type = self.area_type_parents[area_type]
                if _id or area_type is None:
                    break
        return _id, area_type

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
        sort = {
            "_geo_distance": {
                field_name: [x, y],
                "order": order,
                "unit": "km"
            }
        }
        return sort

    @staticmethod
    def get_sort_query(field_name, order="asc"):
        sort = {
            field_name: {
                "order": order,
            }
        }
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

    def search_bool_must_fuzzy_query(self, field_term_dict: dict,
                                     index_name: str,
                                     size: Union[int, None] = None):
        query = self.get_bool_must_fuzzy_query(field_term_dict)
        results = self.es.search(index=index_name, query=query, size=size)
        hits = results['hits']['hits']
        if len(hits) == 0:
            return None
        return hits

    def search_bool_must_fuzzy_query_geo_sort(self,
                                              field_term_dict: dict,
                                              index_name: str,
                                              lat: float, lon: float,
                                              size: Union[int, None] = None):
        assert lat is not None and lon is not None
        query = self.get_bool_must_fuzzy_query(field_term_dict)
        if lat and lon:
            geo_sort = self.get_geo_sort(lat, lon)
            sort = [geo_sort]
        else:
            sort = []
        results = self.es.search(index=index_name, query=query, sort=sort, size=size).body
        hits = results['hits']['hits']
        if len(hits) == 0:
            return None
        return hits

    def get_bool_must_constant_score_query_geo_sort(self,
                                                    field_term_dict: dict,
                                                    index_name: str,
                                                    lat: float, lon: float,
                                                    size: Union[int, None] = None):
        assert lat is not None and lon is not None
        query = self.get_bool_must_constant_score_query(field_term_dict)
        if lat and lon:
            geo_sort = self.get_geo_sort(lat, lon)
            sort = [geo_sort]
        else:
            sort = []
        results = self.es.search(index=index_name, query=query, sort=sort, size=size).body
        hits = results['hits']['hits']
        if len(hits) == 0:
            return None
        return hits

    def search_bool_must_fuzzy_query_range_sort(self,
                                                field_term_dict: dict,
                                                index_name: str,
                                                sort_filed: str, sort_order: str = "asc",
                                                size: Union[int, None] = None):
        assert sort_filed is not None and sort_order is not None
        query = self.get_bool_must_fuzzy_query(field_term_dict)
        if sort_order and sort_filed:
            sort_query = self.get_sort_query(field_name=sort_filed, order=sort_order)
            sort = [sort_query]
        else:
            sort = []
        results = self.es.search(index=index_name, query=query, sort=sort, size=size).body
        hits = results['hits']['hits']
        if len(hits) == 0:
            return None
        return hits

    def search_bool_must_fuzzy_constant_score_range_sort(self,
                                                         field_term_dict: dict,
                                                         index_name: str,
                                                         sort_filed: str, sort_order: str = "asc",
                                                         size: Union[int, None] = None):
        assert sort_filed is not None and sort_order is not None
        query = self.get_bool_must_constant_score_query(field_term_dict)
        if sort_order and sort_filed:
            sort_query = self.get_sort_query(field_name=sort_filed, order=sort_order)
            sort = [sort_query]
        else:
            sort = []
        results = self.es.search(index=index_name, query=query, sort=sort, size=size).body
        hits = results['hits']['hits']
        if len(hits) == 0:
            return None
        return hits
