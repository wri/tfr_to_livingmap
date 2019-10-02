import sys,os 
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from googleapiclient.http import MediaFileUpload
import re
import numpy as np
import pandas as pd
from pyproj import Proj, transform
from affine import Affine
from rasterio.crs import CRS
import rasterio as rio
import tensorflow as tf
from config import COMPRESSION_TYPE, PARALLEL_FILE_READS, PARALLEL_PARSE_CALLS
from config import DEFAULT_STR_VALUE, DEFAULT_NB_VALUE, PATCH_DIMS
import mproc
#
# MAIN
#
def dataset(
        files,
        compression=COMPRESSION_TYPE,
        parallel_reads=PARALLEL_FILE_READS,
        parallel_calls=PARALLEL_PARSE_CALLS):
    dset= _get_dataset(files,compression,parallel_reads)
    return _get_parsed_dataset(dset,parallel_calls)



#
# INTERNAL
#
def _get_dataset(files,compression,parallel_reads):
    if isinstance(files,str):
        files=[files]
    return tf.data.TFRecordDataset(
                filenames=files,
                compression_type=compression,
                num_parallel_reads=parallel_reads)


def _default_value(dtype):
    if dtype==tf.string:
        return DEFAULT_STR_VALUE
    else:
        return DEFAULT_NB_VALUE


def _parse_feature(feat):
    feature_spec={
            b: tf.io.FixedLenFeature(PATCH_DIMS, tf.float32)
            for b in BANDS }
    for key,dtype in FEATURE_PROPS.items():
            feature_spec[key]=tf.io.FixedLenFeature(
                    (), 
                    dtype, 
                    default_value=_default_value(dtype))
    return tf.io.parse_single_example(feat, feature_spec)


def _get_parsed_dataset(dataset,parallel_calls):
    return dataset.map(
            _parse_feature, 
            num_parallel_calls=parallel_calls)






