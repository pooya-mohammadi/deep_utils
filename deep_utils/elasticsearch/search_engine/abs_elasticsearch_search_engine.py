class ElasticSearchABS:
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