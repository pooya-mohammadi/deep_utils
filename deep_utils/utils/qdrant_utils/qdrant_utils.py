from typing import Dict, Any, List, Union, Optional

from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http.models import Record, ScoredPoint


class QdrantUtils:

    def __init__(self, host: Optional[str] = None, port: int = 6333, qdrant_client: Optional[QdrantClient] = None):
        """
        Qdrant normalizes vectors before storing them.
        :param host:
        :param port:
        :param qdrant_client:
        """
        self.host = host
        self.port = port
        self.client = self.client = qdrant_client or QdrantClient(host=self.host, port=self.port)

    def create_collection(self,
                          collection_name: Union[int, str],
                          vector_sizes: List[int],
                          vector_names: List[str]) -> None:
        """
        create a new collection
        :param vector_sizes: size of each vector
        :param vector_names: name of each vector
        :param collection_name: name of the collection
        """
        if len(vector_sizes) != len(vector_names):
            raise ValueError("feature_sizes and vector_names should have the same length")

        vectors_configs = {}
        for vector_size, vector_name in zip(vector_sizes, vector_names):
            vectors_configs[vector_name] = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)

        self.client.recreate_collection(collection_name=collection_name, vectors_config=vectors_configs)

    def add_points(self, collection_name: Union[int, str], point_id, vectors: Dict[str, Any] = None,
                   payloads: Dict[str, Any] = None) -> None:
        """
        add points to a collection_name
        :param collection_name: collection name to be added
        :param vectors: vectors to be added
        :param payloads: payloads to be added
        :param point_id: id to be added
        """
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(point_id),
                    payload=payloads,
                    vector=vectors)
                , ])

    def delete_collection(self, collection_name: Union[int, str]) -> None:
        """
        delete collection
        :param collection_name: collection name to be deleted
        """
        self.client.delete_collection(collection_name=collection_name)
        print("[INFO] Collection Deleted:", collection_name)

    def get_collection_info(self, collection_name: Union[int, str]) -> Dict[str, Any]:
        """
        show collection information overall
        :param collection_name: collection name to be shown
        """

        my_collection_info = self.client.http.collections_api.get_collection(str(collection_name))
        print(my_collection_info.dict())
        return my_collection_info.dict()

    def retrieve_by_id(self, collection_name: Union[int, str], ids: Union[str, List[str]],
                       with_vectors: bool = False) -> List[Record]:
        """
        retrieve by id
        :param collection_name: collection name to be retrieved
        :param ids: id to be retrieved
        :param with_vectors: whether to retrieve vectors
        """
        if isinstance(ids, str):
            ids = [ids]
        return self.client.retrieve(collection_name=collection_name, ids=ids, with_vectors=with_vectors)

    def set_payload(self, collection_name: Union[int, str], payload: Dict[str, Any],
                    points: Union[List[Union[str, int]], Union[str, int]]) -> None:
        """
        set payload
        :param collection_name: collection name to be set
        :param payload: payload to be set
        :param points: points to be set
        """
        self.client.set_payload(collection_name=collection_name,
                                payload=payload,
                                points=points)

    def get_all_collections(self) -> List[str]:
        """
        get all collections
        :return:
        """
        return [collection.name for collection in self.client.http.collections_api.get_collections().result.collections]

    def get_points_in_collection(self, collection_name: Union[str, int], with_payload: bool = True,
                                 with_vector: bool = False) -> \
            List[Record]:  # noqa
        """
        get all points in collection
        :param collection_name: collection name to be retrieved
        :param with_payload: whether to retrieve payload
        :param with_vector: whether to retrieve vector
        :return:
        """
        return self.client.scroll(collection_name=collection_name,
                                  with_payload=with_payload,
                                  with_vectors=with_vector)[0]

    def search(self, collection_name: str,
               vector_name: str,
               features: List[float],
               threshold: Optional[float] = 0.0,
               limit: int = 5) -> List[ScoredPoint]:
        """
        Do a simple research with one vector!
        :param collection_name:
        :param vector_name:
        :param features:
        :param threshold:
        :param limit:
        :return:
        """
        results: List[ScoredPoint] = self.client.search(collection_name=collection_name,
                                                        score_threshold=threshold,

                                                        search_params=models.SearchParams(hnsw_ef=512,
                                                                                          exact=True),
                                                        query_vector=(vector_name, features),
                                                        with_vectors=False,
                                                        append_payload=True,
                                                        limit=limit)
        return results
