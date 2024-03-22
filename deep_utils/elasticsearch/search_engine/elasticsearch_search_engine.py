import os
from typing import Dict, Union, List

from elasticsearch import Elasticsearch, NotFoundError

from deep_utils.utils.json_utils.json_utils import JsonUtils
from deep_utils.utils.logging_utils.logging_utils import value_error_log

from deep_utils.elasticsearch.search_engine.abs_elasticsearch_search_engine import ElasticSearchABS


class ElasticsearchEngin(ElasticSearchABS):
    def __init__(self, elastic_url="http://localhost:9200", es=None, timeout=10, logger=None, verbose=1):
        if es is None:
            self.es = Elasticsearch(elastic_url, timeout=timeout)
        elif isinstance(es, Elasticsearch):
            self.es = es
        else:
            value_error_log(logger, "url or es instance are not provided")
        self.verbose = verbose

    def get_bool_must_match_all_geo_filter(self, index_name, lat, lon, distance, scale="km", size=10):
        """
        GET index-name/_search
{
  "query": {
    "bool": {
      "must": {
        "match_all": {}
      },
      "filter": {
        "geo_distance": {
          "distance": "0.1km",
          "location": {
            "lat": 35.7041126,
            "lon": 51.4629228
          }
        }
      }
    }
  }
}
        :param index_name:
        :param lat:
        :param lon:
        :param distance:
        :param scale:
        :param size:
        :return:
        """
        query = {
            "bool": {
                "must": self.get_match_query(keyword="match_all"),
                "filter": self.get_geo_filter(lat, lon, distance, scale=scale)
            }
        }
        results = self.es.search(index=index_name, query=query, size=size)
        hits = ElasticsearchEngin.get_hits(results)
        return hits

    @staticmethod
    def get_geo_filter(lat, lon, distance, scale="km"):
        filter_ = {
            "geo_distance": {
                "distance": f"{distance}{scale}",
                "location": {
                    "lat": lat,
                    "lon": lon
                }
            }
        }
        return filter_

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
            JsonUtils.dump_json(f"equals/{province}_{city_name}.json", hits, ensure_ascii=False)
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
    def get_bool_must_fuzzy_query(field_term_dict: Dict[str, str]):
        """
        This method is used for getting a must query in fuzzy mode
        GET index-name/_search
        {
        "query":
            {"bool":
                {"must":
                    [{
                    "fuzzy":
                    {"field-name": {"value": "field-value"}}
                    }]
                }
            }
        }

        :param field_term_dict:
        :return:
        """
        must_list = ElasticsearchEngin.get_fuzzy_values(field_term_dict)

        query = {"bool": {"must": must_list}}
        return query

    @staticmethod
    def get_fuzzy_values(field_term_dict):
        """
        Returns the fuzzy values:
        [
         {"fuzzy": {"value_1": {"value": "value"}}},
         {"fuzzy": {"value_1": {"value": "value"}}}
        ]
        :param field_term_dict:
        :return:
        """
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
        return must_list

    def search_bool_must_fuzzy_query(self,
                                     field_term_dict: dict,
                                     index_name: str,
                                     size: Union[int, None] = None):
        """
        This method is used for searching a must query in fuzzy mode with size
        GET index-name/_search
        {
        "query":
            {"bool":
                {"must":
                    [{
                    "fuzzy":
                    {"field-name": {"value": "field-value"}}
                    }]
                }
            },
            "size": size
        }
        :param field_term_dict:
        :param index_name:
        :param size:
        :return:
        """
        query = self.get_bool_must_fuzzy_query(field_term_dict)
        results = self.es.search(index=index_name, query=query, size=size)
        hits = ElasticsearchEngin.get_hits(results)
        return hits

    @staticmethod
    def get_query_multi_match_fuzzy(query_value: str,
                                    field_names: List[str],
                                    fuzziness: Union[int, str] = "AUTO"):
        """
        This method is used to get the query for multi_match which is used for searching a query value in various
        fields. The fuzziness is set to AUTO to compensate for spelling errors
        "query": {
            "multi_match" : {
            "type": best_fields,
              "query":    "query-val",
              "fields": [ "field-1", "field-2" ] ,
              "fuzziness": "AUTO"
            }
        }
        :param query_value:
        :param field_names:
        :param fuzziness: AUTO by default
        :return:
        """

        query = {
            "multi_match": {
                "query": query_value,
                "fields": field_names,
                "fuzziness": fuzziness,
                "type": "best_fields"
            }
        }
        return query

    def search_query_bool_must_multi_match_fuzzy(self,
                                                 index_name,
                                                 multi_match_query_value,
                                                 multi_match_field_names,
                                                 fuzzy_field_term_dict: dict,
                                                 size=None,
                                                 ):
        """
        GET index-name/_search
{
  "query": {
    "bool": {
      "must": [{
        "multi_match": {
          "query": multi_match_query_value,
          "type": "best_fields",
          "fields": multi_match_field_names,
          "fuzziness": "AUTO"
        }
      },
      {
        "fuzzy": {
          "value_1": {"value": "value"}
        }
      },
      {
        "fuzzy": {
          "value_2": {"value": "value"}
        }
      }
      ...
      ]
    }
  }
}
        :param index_name:
        :param multi_match_query_value:
        :param multi_match_field_names:
        :param fuzzy_field_term_dict:
        :param size:
        :return:
        """
        fuzzy_list = self.get_fuzzy_values(fuzzy_field_term_dict)
        multi_match = self.get_query_multi_match_fuzzy(multi_match_query_value, multi_match_field_names)
        query = {
            "bool": {
                "must": [multi_match, *fuzzy_list]
            }
        }
        results = self.es.search(index=index_name, query=query, size=size).body
        hits = self.get_hits(results)
        return hits

    def search_query_multi_match_fuzzy(self,
                                       index_name: str,
                                       query_value: str,
                                       field_names: List[str],
                                       fuzziness: Union[int, str] = "AUTO",
                                       size: Union[int, None] = None):
        """
        This method is used to get the query for multi_match which is used for searching a query value in various
        fields. The fuzziness is set to AUTO to compensate for spelling errors
        GET index-name/_search
        {
            "query": {
                "multi_match" : {
                  "query":    "query-val",
                  "fields": [ "field-1", "field-2" ] ,
                  "fuzziness": "AUTO"
                }
            },
            "size": size
        }
        :param index_name:
        :param query_value:
        :param field_names:
        :param fuzziness: set to AUTO
        :param size:
        :return:
        """
        query = self.get_query_multi_match_fuzzy(query_value=query_value, field_names=field_names, fuzziness=fuzziness)
        results = self.es.search(index=index_name, query=query, size=size)
        hits = ElasticsearchEngin.get_hits(results)
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
        This method is used for query-fuzzy search plus geo sort
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
        GET index_name/_search
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

    def get_match_all_geo_sort_query(self, index_name, lat, lon, sort_field="location", sort_order="asc", size=1):
        query = self.get_match_query(keyword="match_all")
        sort = self.get_geo_sort(lat, lon, field_name=sort_field, order=sort_order)
        results = self.es.search(index=index_name, query=query, sort=sort, size=size).body
        hits = ElasticsearchEngin.get_hits(results)
        return hits

    def search_match_query(self, index_name, field_name="", field_value="", keyword="match", size=None,
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
        results = self.es.search(index=index_name, query=query, size=size).body
        hits = ElasticsearchEngin.get_hits(results, return_source=return_source)
        return hits

    def search_by_id(self, index_name, id_value, id_field_name="_id") -> Union[dict, None]:
        hits = self.search_match_query(index_name, field_name=id_field_name, field_value=id_value, keyword="match",
                                       size=1)
        if len(hits) == 1:
            return hits[0]["_source"]
        else:
            return None

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

    def update_by_id_add_new_field(self, index_name, id_, new_field_name, new_field_value, scripted_upsert=False,
                                   detect_noop=False):
        """
        This adds a new field to an existing document!
        POST test/_update/1
        {
          "script" : "ctx._source.new_field = 'value_of_new_field'"
        }


        :param index_name:
        :param id_:
        :param new_field_name:
        :param new_field_value:
        :param scripted_upsert:
        :param detect_noop:
        :return:
        """
        script = f"ctx._source.{new_field_name} = '{new_field_value}'"
        result = self.es.update(index=index_name,
                                id=id_,
                                detect_noop=detect_noop,
                                scripted_upsert=scripted_upsert,
                                script=script)
        return result

    def update_by_id_remove_field(self, index_name, id_, remove_field_name, scripted_upsert=False,
                                  detect_noop=False):
        """
        This adds a new field to an existing document!
        POST test/_update/1
        {
            "script" : "ctx._source.remove('new_field')"
        }

        :param index_name:
        :param id_:
        :param remove_field_name:
        :param scripted_upsert:
        :param detect_noop:
        :return:
        """
        script = f"ctx._source.remove('{remove_field_name}')"
        result = self.es.update(index=index_name,
                                id=id_,
                                detect_noop=detect_noop,
                                scripted_upsert=scripted_upsert,
                                script=script)
        return result

    def check_index_exists(self, index_name, create=False, es_mapping=Union[dict, None]):
        """
        Checks whether this index exists or not
        :param index_name:
        :param create: If set to true creates the index
        :param es_mapping:
        :return:
        """
        try:
            self.es.get(index=index_name, id="an id that might not be chosen by someone!")
        except NotFoundError as e:
            if e.message == "index_not_found_exception":
                if create:
                    self.es.indices.create(index=index_name, mappings=es_mapping)
                    # After being created return True:)
                    return True
                return False
        return True
