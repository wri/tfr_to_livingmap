import tensorflow as tf
#
# PROPERTIES
#
FEATURE_PROPS={
        'tile_id': tf.string,
        'date': tf.string,
        'crs': tf.string,
        'lon': tf.float32,
        'lat': tf.float32,
        'cirrus_score': tf.float32,
        'opaque_score': tf.float32,
        'black_score': tf.float32,
        'centroid_lc_type': tf.float32,
        'BIOME_NUM': tf.float32,
        'BIOME_NAME': tf.string,
        'ECO_NAME': tf.string,
        'ECO_ID': tf.float32,
        'country_na': tf.string,
        'wld_rgn': tf.string
}
INPUT_BANDS=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
RGB_BANDS=['red','green','blue']
CLOUD_BANDS=['cirrus','opaque']
BANDS=INPUT_BANDS+RGB_BANDS+CLOUD_BANDS
#
# IMAGE CONFIG
#
SIZE=510
HALF_SIZE=SIZE//2
RES=10
PATCH_DIMS=[SIZE, SIZE]
#
# GCS/RUN
#
BUCKET='living-map-dev'
FOLDER='tmp'
NOISY=False
NOISE_REDUCER=100
#
# PARSER
#
COMPRESSION_TYPE='GZIP'
PARALLEL_FILE_READS=5
PARALLEL_PARSE_CALLS=2
DEFAULT_STR_VALUE=''
DEFAULT_NB_VALUE=0