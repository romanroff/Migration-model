REGION_NAME = "Ленинградская область, Россия"
MODEL_PATH = "models/RF/rf_model.joblib"
API_OVERLAP_MODEL_PATH = "models/RF/rf_model_api_overlap.joblib"
DATA_PATH = "data"
PLACE_FILTERS = {"place": ["city", "town", "village"]}
FEATURE_COLS = ['d', 'm_o', 'm_d']
API_OVERLAP_FEATURE_COLS = [
    'd',
    'area_o',
    'area_d',
    'm_o',
    'm_d',
    'health_point_o',
    'health_point_d',
    'main_road_line_o',
    'main_road_line_d',
    'school_point_o',
    'school_point_d',
]
MAIN_INFO_API_URL = "http://10.32.1.47:5000/api/migrations/main_info"
MAIN_INFO_TIMEOUT = 180
LOOP_REQUEST_SLEEP_SECONDS = 0.1
TOP_PERCENT = 1
