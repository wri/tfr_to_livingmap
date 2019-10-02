import sys,os 
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import re
import numpy as np
import pandas as pd
from pyproj import Proj, transform
from affine import Affine
from rasterio.crs import CRS
import rasterio as rio
import tensorflow as tf
import mproc
from config import FEATURE_PROPS, INPUT_BANDS, RGB_BANDS, CLOUD_BANDS, BANDS
from config import SIZE, HALF_SIZE, RES, BUCKET, FOLDER, NOISY, NOISE_REDUCER
from config import FOLDER, BUCKET
#
# MAIN
#
def run(
    run_name,
    dataset,
    gcs_service=None,
    folder=FOLDER,
    bucket=BUCKET,
    take=5,
    skip=None):
    rows=[]
    if not gcs_service:
        gcs_service=build('storage', 'v1')
    for i,element in enumerate(dataset.skip(skip).take(take)):
        if NOISY and (not (i%NOISE_REDUCER)): 
                print(i,'...')
        props=properties(element)
        im_dict={b: element[b].numpy() for b in BANDS}
        inpt=_get_image(im_dict,INPUT_BANDS,np.uint16)
        rbg=_get_image(im_dict,RGB_BANDS,np.uint8)
        profile=get_profile(
                props['lon'],
                props['lat'],
                props['crs'])
        tile_id=props['tile_id']
        date=props['date']
        date=re.sub('-','',date)
        _tif_to_gcs(
            gcs_service,
            inpt,
            profile,
            tile_id,
            date
            folder,
            bucket)
        _png_to_gcs(
            gcs_service,
            rbg,
            profile,
            tile_id,
            date
            folder,
            bucket)      
        rows.append(props)
    df=pd.DataFrame(rows)
    df.to_csv('tmp.csv')
    to_gcs(
        gcs_service=gcs_service,
        src='tmp.csv',
        dest=run_name,
        mtype='text/csv',
        folder=f'{folder}/CSV')
    print(f'log: gs://{bucket}/{folder}/CSV/{run_name}')




#
# UTILS
#
def input_image(im_dict):
    im=np.stack([im_dict[k] for k in INPUT_BANDS])
    return im.astype(np.uint16)


def rgb_image(im_dict):
    im=np.stack([im_dict[k] for k in RGB_BANDS])
    return im.astype(np.uint8)


def properties(elm):
    jsn={ k: _clean(tf.get_static_value(elm[k])) for k in FEATURE_PROPS }
    return jsn


def get_profile(lon,lat,crs,size=SIZE):
    x,y=transform(Proj(init='epsg:4326'),Proj(init=crs),lon,lat)
    x,y=int(round(x)),int(round(y))
    xmin=x-HALF_SIZE
    ymin=y-HALF_SIZE
    return {
            'compress': 'lzw',
            'count': len(INPUT_BANDS),
            'crs': CRS.from_dict(init=crs),
            'driver': 'GTiff',
            'dtype': 'uint16',
            'height': SIZE,
            'interleave': 'pixel',
            'nodata': None,
            'tiled': False,
            'transform': Affine(RES,0,xmin,0,-RES,ymin),
            'width': SIZE }


def to_gcs(
        gcs_service,
        src,
        dest,
        mtype,
        folder=None,
        bucket=BUCKET):
    media = MediaFileUpload(
            src, 
            mimetype=mtype,
            resumable=True)
    if folder:
        dest='{}/{}'.format(folder,dest)
    request=gcs_service.objects().insert(bucket=bucket, 
        name=dest,
        media_body=media)
    response=None
    while response is None:
        _, response=request.next_chunk()
    return request, response




#
# INTERNAL
#
def _get_image(im_dict,keys,dtype):
    im=np.stack([im_dict[k] for k in keys])
    return im.astype(dtype)


def _tif_to_gcs(gcs_service,im,profile,tile_id,date,folder=FOLDER,bucket=BUCKET):
        """ write image
        Args: 
                - im<np.array>: image
                - path<str>: destination path
                - profile<dict>: image profile
                - makedirs<bool>: if True create necessary directories
        """  
        with rio.open('tmp.tif','w',**profile) as dst:
                dst.write(im)
        to_gcs(
                gcs_service=gcs_service,
                src='tmp.tif',
                dest=f'{tile_id}-{date}.tif',
                mtype='image/tiff',
                folder=f'{folder}/S2',
                bucket=bucket)


def _png_to_gcs(gcs_service,im,profile,tile_id,date,folder=FOLDER,bucket=BUCKET):
        profile=profile.copy()
        profile['count']=len(RGB_BANDS)
        profile['driver']='PNG'
        profile['dtype']='uint8'
        profile.pop('compress')
        profile.pop('interleave')
        profile.pop('tiled')
        with rio.open('tmp.png','w',**profile) as dst:
            dst.write(im)
        to_gcs(
            gcs_service=gcs_service,
            src='tmp.png',
            dest=f'{tile_id}-{date}.png',
            mtype='image/png',
            folder=f'{folder}/RGB',
            bucket=bucket)



def _clean(value):
    if isinstance(value,bytes):
            value=value.decode("utf-8")
    return value

