class KafkaUtils:
    @staticmethod
    def create_topic(*topics, admin_client=None, bootstrap_servers="localhost:9092", num_partitions=1,
                     replication_factor=1, validate_only=False, client_id="kafka_utils",
                     logger=None, verbose=1):
        """
        Simply creates input topics
        :param topics:
        :param admin_client:
        :param bootstrap_servers:
        :param num_partitions:
        :param replication_factor:
        :param validate_only:
        :param client_id:
        :param logger:
        :param verbose:
        :return:
        """
        from kafka.admin import NewTopic
        from kafka.errors import TopicAlreadyExistsError
        from deep_utils.utils.logging_utils.logging_utils import log_print
        admin_client = KafkaUtils.create_admin_client(admin_client, bootstrap_servers, client_id)
        topic_list = [NewTopic(name=topic, num_partitions=num_partitions, replication_factor=replication_factor) for
                      topic in topics]
        try:
            admin_client.create_topics(new_topics=topic_list, validate_only=validate_only)
            log_print(logger, f"Successfully created {topics}", verbose=verbose)
        except TopicAlreadyExistsError:
            log_print(logger, f"Topics: {topics} already exist", verbose=verbose)

    @staticmethod
    def create_admin_client(admin_client=None, bootstrap_servers="localhost:9092", client_id="kafka_utils"):
        if admin_client is None:
            from kafka.admin import KafkaAdminClient
            admin_client = KafkaAdminClient(
                bootstrap_servers=bootstrap_servers,
                client_id=client_id,
                api_version=(0, 9)
            )
        return admin_client

    @staticmethod
    def create_producer(bootstrap_servers):
        from kafka import KafkaProducer
        producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        return producer

    @staticmethod
    def check_kafka_status(admin_client=None, bootstrap_servers="localhost:9092", status_key="KAFKA_STATUS"):
        admin_client = KafkaUtils.create_admin_client(admin_client, bootstrap_servers=bootstrap_servers)
        try:
            topics = admin_client.list_topics()
            if topics:
                return {status_key: "Alive"}
        except:
            pass
        return {status_key: "DOWN"}
