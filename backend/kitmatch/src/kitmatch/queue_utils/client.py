import pika
from kitmatch.queue_utils import QueueMessage, MessageType
import json
import os
import logging
from kitmatch.config import Settings
from kitmatch.config import RABBIT_BLOCKED_CONN_TIMEOUT_MODELS, RABBIT_HEARTBEAT_MODELS

class QueueClient:
    def __init__(self):
        self.connection = None
        self.channel = None
        
        settings = Settings()
        self.queue_host = settings.rabbitmq_host
        self.message = None
        
    def create_connections(self):
        if self.connection is None and self.channel is None:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(self.queue_host, heartbeat=RABBIT_HEARTBEAT_MODELS, blocked_connection_timeout=RABBIT_BLOCKED_CONN_TIMEOUT_MODELS)
            )
            self.channel = self.connection.channel()
            self.channel.basic_qos(prefetch_count=0)

    def send_message(self, message, queue='models'):
        self.message = message
        msg_json = message.to_json()
        self.channel.basic_publish(exchange='',
                                   routing_key=queue,
                                   body=msg_json,
                                   properties=pika.BasicProperties(
                                       delivery_mode=2,  # make message persistent
                                   ))

    def _process_message(self, channel, method, properties, body):
        message = json.loads(body)
        message = QueueMessage.from_json(message)
        if self.message.id == message.id:
            self.message = message
            channel.basic_ack(delivery_tag=method.delivery_tag)
            channel.stop_consuming()
            logging.debug(f"    • [Client] Message processed — {message.msg_type.name} for Task={message.task_id} | Image={message.image_id}")

    def listen(self):
        self.channel.basic_consume(queue='result', on_message_callback=self._process_message)
        self.channel.start_consuming()
        return self.message


if __name__ == "__main__":
    client = QueueClient()
    msg = QueueMessage(
        task_id="test_task_id",
        image_id="test_image_id",
        id="test_message_id",
        msg_type=MessageType.object_detection,
        data={"test": "data"}
    )

    client.message = msg
    client.create_connections()
    client.listen()