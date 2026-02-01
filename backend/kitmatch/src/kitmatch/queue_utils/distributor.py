import pika
import threading
import json
import os
import time
import requests
import codecs
import pickle
import uuid
from loguru import logger

from kitmatch.config import Settings
from kitmatch.queue_utils import QueueMessage, MessageType
from kitmatch.db.detection_storage import DetectionStorage
from kitmatch.config import RABBIT_HEARTBEAT_MODELS, RABBIT_BLOCKED_CONN_TIMEOUT_MODELS

class ModelsInferenceDistributor:
    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        logger.debug(f"num threads = {self.num_threads}")
        self.threads = []
        self.connection = None
        self.channel = None
        
        settings = Settings()
        self.queue_host = settings.rabbitmq_host
        logger.debug(f"rabbit host = {self.queue_host}")
        self.database = DetectionStorage()

    def start(self):
        logger.info(f"Starting {self.__class__.__name__}...")
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(self.queue_host, heartbeat=RABBIT_HEARTBEAT_MODELS, blocked_connection_timeout=RABBIT_BLOCKED_CONN_TIMEOUT_MODELS)
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare("models")
        self.channel.queue_declare("postprocessing")
        self.channel.queue_declare("result")
        
        self.channel.basic_qos(prefetch_count=1)

        for _ in range(self.num_threads):
            thread = threading.Thread(target=self._consume_messages)
            thread.start()
            self.threads.append(thread)

    def _process_message(self, channel, method, properties, body):
        message = None
        try:
            message = json.loads(body)
            message = QueueMessage.from_json(message)
            ctx_logger = logger.bind(task_id=message.task_id, image_id=message.image_id)
            ctx_logger.info(f"► [Models] Получена задача '{message.msg_type.name}'")
            
            # Выполняем инференс модели
            from kitmatch.utils.utils import modelsRegister
            ctx_logger.info("  › [Models] Начало инференса модели")
            start_time = time.time()
            result = modelsRegister.get_model(message.msg_type).model_inference(message.data)
            inference_time = time.time() - start_time
            ctx_logger.info(f"  ✔ [Models] Инференс завершен за {inference_time:.4f} сек.")
            
            message.data.update(result)
        except Exception as e:
            err_logger = logger.bind(task_id=getattr(message, 'task_id', 'N/A'), image_id=getattr(message, 'image_id', 'N/A'))
            err_logger.error(f"✖ [Models] Ошибка при обработке сообщения: {e}")
            if message:
                for key, value in message.data.items():
                    err_logger.debug(f"    • [Models] data[{key}] type={type(value)}")
                message.data = {"error": str(e)}
                try:
                    self.database.set_detection_error(message.task_id, message.image_id, str(e))
                except requests.exceptions.RequestException as e2:
                    err_logger.error(f"✖ [Models] Не удалось сохранить ошибку в БД: {e2}")
            else:
                logger.error("✖ [Models] Не удалось даже распарсить сообщение")
                message = QueueMessage(task_id='parse_error', image_id='parse_error', id=str(uuid.uuid4()), msg_type=MessageType.object_detection, data={"error": "Message parsing failed"})

        # Убираем изображение из данных перед отправкой на пост-обработку
        if 'image' in message.data:
            del message.data['image']

        msg_json = message.to_json()

        channel.basic_publish(
            exchange='',
            routing_key='postprocessing',
            body=msg_json,
        )
        channel.basic_ack(delivery_tag=method.delivery_tag)
        logger.bind(task_id=message.task_id, image_id=message.image_id).info("  • [Models] Задача отправлена в 'postprocessing'")

    def _consume_messages(self):
        self.channel.basic_consume(queue='models', on_message_callback=self._process_message)
        self.channel.start_consuming()

    def stop(self):
        logger.info("› [Models] ◀ Stopping distributor…")
        for _ in range(self.num_threads):
            self.channel.stop_consuming()
        for thread in self.threads:
            logger.info(f"Stopping thread: {thread.name}")
            thread.join()

        self.connection.close()
        logger.info("✔ [Models] Connection closed")


class PostProcessingDistributor(ModelsInferenceDistributor):
    def __init__(self, num_threads=1):
        super().__init__(num_threads)

    def _process_message(self, channel, method, properties, body):
        message = None
        try:
            message = json.loads(body)
            message = QueueMessage.from_json(message)
            ctx_logger = logger.bind(task_id=message.task_id, image_id=message.image_id)
            ctx_logger.info(f"► [PostProc] Получена задача '{message.msg_type.name}'")
            
            if "error" in message.data:
                ctx_logger.warning("  ! [PostProc] В задаче есть ошибка, пропускается обработка, отправляется дальше.")
            elif message.msg_type != MessageType.results:
                from kitmatch.utils.utils import modelsRegister
                ctx_logger.info("  › [PostProc] Начало финальной обработки")
                result = modelsRegister.get_model(message.msg_type).get_main_result(message.data)
                
                if result is None:
                    ctx_logger.warning("  ⚠️ [PostProc] Финальная обработка не вернула результата")
                    message.data = {"error": "something wrong"}
                else:
                    message.data.update(result)
                    ctx_logger.info("  ✔ [PostProc] Финальная обработка завершена")
                    # Сохраняем результаты в БД
                    ctx_logger.info("  › [PostProc] Сохранение результата в MongoDB")
                    self.database.create_detection_result(message.task_id, message.image_id)
                    self.database._set_info(message.task_id, message.image_id, message.data)
                    ctx_logger.info("  ✔ [PostProc] Результат успешно сохранен в MongoDB")

        except Exception as e:
            err_logger = logger.bind(task_id=getattr(message, 'task_id', 'N/A'), image_id=getattr(message, 'image_id', 'N/A'))
            err_logger.error(f"✖ [PostProc] Ошибка при обработке сообщения: {e}")
            if message:
                for key, value in message.data.items():
                    err_logger.debug(f"    • data[{key}] type={type(value)}")
                message.data = {"error": str(e)}
                try:
                    self.database.set_detection_error(message.task_id, message.image_id, str(e))
                except requests.exceptions.RequestException as e2:
                    err_logger.error(f"✖ [PostProc] Не удалось сохранить ошибку в БД: {e2}")
            else:
                logger.error("✖ [PostProc] Не удалось даже распарсить сообщение")
                message = QueueMessage(task_id='parse_error', image_id='parse_error', id=str(uuid.uuid4()), msg_type=MessageType.object_detection, data={"error": "Message parsing failed"})

        msg_json = message.to_json()
        channel.basic_publish(
            exchange='',
            routing_key='result',
            body=msg_json,
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
        ))

        channel.basic_ack(delivery_tag=method.delivery_tag)
        logger.bind(task_id=message.task_id, image_id=message.image_id).info("  • [PostProc] Результат отправлен в очередь 'result'")

    def _consume_messages(self):
        self.channel.basic_consume(queue='postprocessing', on_message_callback=self._process_message)
        self.channel.start_consuming()

if __name__ == "__main__":
    models_distributor = ModelsInferenceDistributor(1)
    postproc_distributor = PostProcessingDistributor(1)
    
    try:
        models_distributor.start()
        postproc_distributor.start()
    except Exception as e:
        logger.error(f"Error starting distributors: {e}")
        models_distributor.stop()
        postproc_distributor.stop()
