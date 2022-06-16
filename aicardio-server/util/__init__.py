# coding=utf-8

import logging
import math
import json
from flask import current_app as app

__author__ = 'Tung.Luu'
_logger = logging.getLogger(__name__)


video_exts = ['mp4']
image_exts = ['jpeg', 'png', 'jpg']


def filter_attr(obj, attr):
    return {key: obj[key] for key in (attr & obj.keys())}


def remove_attr(obj, rm_attr):
    return {k: v for k, v in obj.items() if k not in rm_attr}


def valid_req(request, comp_attr=[], ext_attr=[]):
    """ Validate a json request and its attributes
        :param obj request
        :param [str] comp_attr: compulsory attributes
        :param [str] ext_attr: extended attributes
    """
    try:
        data = request.data
        if isinstance(data, bytes):
            data = data.decode()
        req_loaded = json.loads(data)
    except Exception as err:
        raise ValueError(
            'Cannot parse request, request format must be application/json!')
    req_attr = req_loaded.keys()
    if not set(req_attr) >= set(comp_attr):
        raise ValueError('Request must have: ' + ', '.join(comp_attr))
    req_filtered = filter_attr(req_loaded, comp_attr + ext_attr)
    return req_filtered


def allowed_filename(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


def paginate(data, current_page=1, per_page=20):
    offset = (current_page - 1) * per_page
    n_pages = math.ceil(len(data) / per_page)
    pages = [current_page - 2, current_page - 1,
             current_page, current_page + 1, current_page + 2]
    pages = list(filter(lambda x: 1 < x and x < n_pages, pages))
    if not len(pages):
        pages = list(range(1, n_pages+1))
    else:
        pages = [1] + ([] if pages[0] == 2 else [None]) + pages
        pages = pages + ([] if pages[-1] == n_pages-1 else [None]) + [n_pages]
    return data[offset:offset+per_page], current_page, pages
