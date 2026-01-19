from model.tft_model import TFTPriceModel

_MODEL_CACHE = {}

def get_model(zone: str) -> TFTPriceModel:
    if zone not in _MODEL_CACHE:
        _MODEL_CACHE[zone] = TFTPriceModel.load(zone)
    return _MODEL_CACHE[zone]
