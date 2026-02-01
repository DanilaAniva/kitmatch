import codecs
import pickle
import os
import numpy as np

class SecondDbClient():
    def __init__(self, host=None, port=None):
        self._client = None
        self.host, self.port = host, port
    
    def __connect(self):
        pass

    def __disconnect(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def save_pickle(self, task_id: str, image_id: str, data: np.ndarray, name: str) -> str:
        directory = "./detection_pickles"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory_2 = f"./detection_pickles/{task_id}"
        if not os.path.exists(directory_2):
            os.makedirs(directory_2)

        path = f"{directory_2}/{image_id}_{name}"
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return path
    
    def get_from_pickle(self, task_id: str, image_id: str, path: str):
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                encoded_data = codecs.encode(
                    pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL), "base64"
                ).decode("latin1")
            return data
    