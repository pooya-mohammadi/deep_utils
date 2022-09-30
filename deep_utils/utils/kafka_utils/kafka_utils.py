def create_producer(bootstrap_servers):
    from kafka import KafkaProducer
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
    return producer

