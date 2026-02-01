import uuid
import pickle

from .client import QueueClient
from .message import QueueMessage, MessageType

def save_message_to_pickle(message, filename):
    with open(f"kitmatch/src/kitmatch/msgs_pickle_cache/{filename}", 'wb') as f:
        pickle.dump(message, f)
    print(f"Message saved to {filename}")


def send_mes_to_queue(
    task_id: str, image_id: str, data: dict, msg_type: MessageType, queue: str = "models"
):
    id = str(uuid.uuid4())
    message = QueueMessage(
        task_id=task_id,
        image_id=image_id,
        id=id,
        msg_type=msg_type,
        data=data,
    )
    client = QueueClient()
    client.create_connections()
    client.send_message(message, queue=queue)
    result = client.listen()
    if "error" in result.data:
        error_message = result.data["error"]
        if len(error_message) > 1000:
            print(f"Error from queue: {error_message[:1000]}...")
            raise Exception(f"Error from {queue} queue: {error_message[:1000]}...")
        else:
            print(f"Error from queue: {error_message}")
            raise Exception(f"Error from {queue} queue: {error_message}")
    return result
