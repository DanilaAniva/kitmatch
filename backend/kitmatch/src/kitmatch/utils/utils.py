from kitmatch.queue_utils import MessageType

def singleton(class_):
    """
    Singleton decorator
    :param class_: class
    :return: class instance
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return get_instance


@singleton
class ModelRegister(object):
    def __init__(self):
        self._models_cls = dict()
        from kitmatch.config import Settings
        self.settings = Settings()

    def register(self, message_type: MessageType, triton_service_name):
        def _register(cls):
            self._models_cls[message_type] = cls(host=self.settings.triton_url, service_name=triton_service_name)
            return cls
        return _register

    def multi_register(self, message_types, triton_service_name):
        def _decorator(cls):
            for _mt in message_types:
                self.register(message_type=_mt, triton_service_name=triton_service_name)(cls)
            return cls
        return _decorator

    def get_model(self, message_type: MessageType):
        return self._models_cls[message_type]


modelsRegister = ModelRegister()