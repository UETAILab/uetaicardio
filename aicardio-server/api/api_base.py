# coding=utf-8
import logging
from flask_restplus import fields
from api import api
from helpers.error_handler_helper import HandledError

__author__ = 'Tung.Luu'
_logger = logging.getLogger(__name__)


GENERAL_RESP = api.model('GENERAL_RESP', {
    'error': fields.Integer(),
    'data': fields.String()
})


class BaseApi(object):
    @staticmethod
    def api_response(http_code=200, data=None, error=None, handled_error=None):
        """ Return http response
        :param str error: error message, null if no error
        :param obj data: response data
        :param int http_code:
        """
        if isinstance(handled_error, HandledError):
            error, http_code = handled_error.message, handled_error.error_code
        elif http_code == 500:
            error = error or 'Internal server error!'
        elif http_code == 200:
            data = data or 'Success'
        return (
            {'error': error, 'data': data},
            http_code
        )
