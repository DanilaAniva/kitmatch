import enum
import json
import numpy as np
from PIL import Image
import base64
import codecs
import pickle


class MessageType(enum.Enum):
    # Типы сообщений для детекции инструментов
    input_image = 0          # Входное изображение для обработки
    object_detection = 1     # Детекция объектов
    postprocessing = 2       # Постобработка результатов
    results = 3              # Финальные результаты
    save_results = 4         # Сохранение результатов в БД
    compose_final = 5        # Композиция финального результата

class QueueMessage:
    def __init__(self, task_id: str, image_id: str, id: str, msg_type: MessageType, data: dict):
        self.task_id = task_id
        self.image_id = image_id
        self.id = id
        self.msg_type = msg_type
        self.data = data

    def _prepare_for_serialize(self):
        data = {
            "id": self.id,
            "task_id": self.task_id,
            "image_id": self.image_id,
            "msg_type": self.msg_type.value,
            "data": {}
        }
        for key in self.data.keys():
            if isinstance(self.data[key], np.ndarray):
                obj_base64string = codecs.encode(pickle.dumps(self.data[key], protocol=pickle.HIGHEST_PROTOCOL),
                                                 "base64").decode('latin1')
                d = {"data": obj_base64string, "type": str(np.ndarray)}
                data["data"][key] = d
            elif isinstance(self.data[key], Image.Image):
                tmp = np.array(self.data[key])
                obj_base64string = codecs.encode(pickle.dumps(tmp, protocol=pickle.HIGHEST_PROTOCOL),
                                                 "base64").decode('latin1')
                d = {"data": obj_base64string, "type": str(Image.Image)}
                data["data"][key] = d
            else:
                d = {"data": self.data[key]}
                data["data"][key] = d
        return data

    def to_json(self):
        data = self._prepare_for_serialize()
        return json.dumps(data)

    @staticmethod
    def from_json(json_msg):
        msg_type = MessageType(json_msg["msg_type"])
        _id = json_msg["id"]
        image_id = json_msg["image_id"]
        task_id = json_msg["task_id"]

        data = {}
        for key, value in json_msg["data"].items():
            if len(value.keys()) == 1:
                data[key] = value["data"]
                continue
            d, type = value["data"], value["type"]
            if type in [str(np.ndarray), str(Image.Image)]:
                d = pickle.loads(codecs.decode(d.encode('latin1'), "base64"))
            if type == str(Image.Image):
                d = Image.fromarray(d)
            data[key] = d
        return QueueMessage(task_id=task_id, image_id=image_id, id=_id, msg_type=msg_type, data=data)


# if __name__ == "__main__":
#     msg = QueueMessage(msg_type=MessageType.input, data={
#         "image": np.random.randn(224, 224, 3).astype(np.uint8),
#         "mask": np.random.randn(224, 244).astype(bool)
#     })
#     msg_json = json.loads(msg.to_json())
#     msg1 = QueueMessage.from_json(msg_json)
#     print()
