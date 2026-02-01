from kitmatch.utils.utils import modelsRegister
from kitmatch.queue_utils import MessageType

def register_executor(message_type: MessageType, triton_service_name: str):
    return modelsRegister.register(message_type, triton_service_name)


