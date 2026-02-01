from abc import abstractmethod


class BaseExecutor:
    def __init__(self, host, service_name):
        self.service_name = service_name
        self.host = host

    @abstractmethod
    def get_model_result(self, data):
        return data

    @abstractmethod
    def model_inference(self, data):
        result = self.get_model_result(data)
        return result

    @abstractmethod
    def get_main_result(self, data):
        return data