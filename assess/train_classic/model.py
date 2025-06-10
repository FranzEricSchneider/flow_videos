from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


MODELS = {
    "forest": RandomForestRegressor,
    "ridge": Ridge,
    "SVR": SVR,
}


def get_model(key: str, scale: bool):
    model = MODELS[key]()
    if scale is True:
        return make_pipeline(StandardScaler(), model)
    else:
        return model
