import pika
import threading
import json
import logging
import uuid
import os

from kitmatch.config import Settings, RABBIT_BLOCKED_CONN_TIMEOUT_REQUESTS, RABBIT_HEARTBEAT_REQUESTS
from kitmatch.queue_utils import QueueMessage, MessageType
from kitmatch.db.detection_storage import DetectionStorage

ACTION_MAP = {
    "process": None,  # Будет определено позже
    "regenerate": None,
    "retry": None,
}

class RequestDistributor:
    def __init__(self):
        self.connection = None
        self.channel = None
        
        settings = Settings()
        self.queue_host = settings.rabbitmq_host
        self.is_running = False
        
    def start(self):
        logging.info("[QUEUE] Starting Request Distributor...")
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    self.queue_host, 
                    heartbeat=RABBIT_HEARTBEAT_REQUESTS, 
                    blocked_connection_timeout=RABBIT_BLOCKED_CONN_TIMEOUT_REQUESTS
                )
            )
            self.channel = self.connection.channel()
            
            # Создаем очередь
            self.channel.queue_declare(queue='processing', durable=True)
            
            # prefetch_count=1 для строгого FIFO
            self.channel.basic_qos(prefetch_count=1)
            
            # Запускаем один поток обработки
            self.is_running = True
            thread = threading.Thread(target=self._consume_messages)
            thread.daemon = True
            thread.start()
            
            logging.info("[QUEUE] Request Distributor started")
            
        except Exception as e:
            logging.error(f"[QUEUE] Failed to start Request Distributor: {e}")
            raise
    
    def _process_message(self, channel, method, properties, body):
        message = QueueMessage.from_json(json.loads(body))
        action  = message.data.get("action")
        payload = message.data.get("request", {})
        debug    = message.data.get("debug", False)
        local_storage = DetectionStorage()

        try:
            handler = ACTION_MAP.get(action, None)
            if handler is None:
                logging.warning(f"[QUEUE] Unknown action: {action}")
            else:
                # вызываем функцию-обработчик
                handler(payload, local_storage, debug)
                logging.debug(f"[QUEUE] Completed action={action}, task={message.task_id}")
        except Exception as e:
            logging.error(f"[QUEUE] Error on action={action}, task={message.task_id}: {e}")
        finally:
            # в любом случае ACK, чтобы не блокировать очередь
            channel.basic_ack(delivery_tag=method.delivery_tag)
    
    def _consume_messages(self):
        """Потребление сообщений из FIFO очереди"""
        try:
            self.channel.basic_consume(
                queue='processing', 
                on_message_callback=self._process_message
            )
            logging.info("[QUEUE] Started consuming messages...")
            self.channel.start_consuming()
        except Exception as e:
            logging.error(f"[QUEUE] Error in message consumption: {e}")
    
    def stop(self):
        """Остановка FIFO обработчика"""
        logging.info("[QUEUE] Stopping distributor...")
        self.is_running = False
        if self.channel:
            self.channel.stop_consuming()
        if self.connection:
            self.connection.close()
        logging.info("[QUEUE] Distributor stopped")



def send_task_to_queue(
    action: str,
    request_data: dict,
    debug: bool = False
):
    try:
        # Подключение к RabbitMQ
        settings = Settings()
        queue_host = settings.rabbitmq_host
            
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                queue_host, 
                heartbeat=RABBIT_HEARTBEAT_REQUESTS, 
                blocked_connection_timeout=RABBIT_BLOCKED_CONN_TIMEOUT_REQUESTS
            )
        )
        channel = connection.channel()
        
        # Убеждаемся что очередь существует
        channel.queue_declare(queue='processing', durable=True)
        
        message = QueueMessage(
            task_id=request_data.get("task_id", str(uuid.uuid4())),
            image_id=request_data.get("image_id", ""),
            id=str(uuid.uuid4()),
            msg_type=MessageType.input_image,
            data={
                "action": action,
                "request": request_data,
                "debug": debug
            }
        )
        
        channel.basic_publish(
            exchange='',
            routing_key='processing',
            body=message.to_json(),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Делаем сообщение persistent
            )
        )
        
        connection.close()
        
        logging.debug(f"[QUEUE] Task queued {action} — Task={request_data.get('task_id')}")
        return True
        
    except Exception as e:
        logging.error(f"[QUEUE] Failed to queue task: {e}")
        raise
