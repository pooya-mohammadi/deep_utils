try:
    from deep_utils.dummy_objects.elasticsearch import ElasticsearchEngin
    from .elasticsearch_search_engine import ElasticsearchEngin
except ModuleNotFoundError:
    pass
