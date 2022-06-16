import logging
import urllib
from flask import request

from services import dicom_service
from view.admin import app, jinja_env
from view.admin.auth import authorize
from view.routes import routing_table
import util

_logger = logging.getLogger(__name__)


@app.route(routing_table['admin']['dashboard'], methods=['GET'])
@authorize
def dashboard():
    dashboard_template = jinja_env.get_template('dashboard.html')
    # sort by date
    date_order = request.args.get('date_order', 'desc', type=str)
    # pagination
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    per_page = min(per_page, 50)
    # filters
    device = request.args.get('device', None, type=str)
    confirm = request.args.get('confirm', None, type=int)
    dicom_avai = request.args.get('dicom_avai', 0, type=int)
    n_chambers = request.args.get('n_chambers', None, type=str)
    # load stat data
    overview, by_date_and_user = dicom_service.stat.stat_on_folder()
    for row in overview:
        if row['nchamber'] in ['LABEL', '-1']:
            row['nchamber'] = None
    device_list = list(set([x['device'] for x in overview]))
    extract_url_and_filename(overview)
    if dicom_avai:
        overview = list(filter(lambda x: x['gif_url'] is not None, overview))
    if device:
        overview = list(filter(lambda x: x['device'] == device, overview))
    if n_chambers == '-1':
        overview = list(filter(lambda x: x['nchamber'] is None, overview))
    elif n_chambers:
        overview = list(
            filter(lambda x: x['nchamber'] == n_chambers, overview))
    if confirm is not None:
        check_file_confirmation(overview)
        overview = list(filter(lambda x: x['confirm'] == confirm, overview))
    # sort and paginate
    overview.sort(key=lambda x: x['date'], reverse=(date_order == 'desc'))
    by_date_and_user.sort(
        key=lambda x: x['date'], reverse=(date_order == 'desc'))
    overview, current_page, pages = util.paginate(overview, page, per_page)
    if confirm is None:
        check_file_confirmation(overview)
    return dashboard_template.render({
        'data': (overview, by_date_and_user),
        'selectable_data': dict({
            'devices': device_list,
        }),
        'filters': dict({
            'device': device,
            'confirm': confirm,
            'dicom_avai': dicom_avai,
            'n_chambers': n_chambers,
        }),
        'pagination': dict({
            'current_page': current_page,
            'per_page': per_page,
            'pages': pages,
        }),
        'show_details': request.args.get('show_details', False, type=bool),
        'date_order': date_order,
    })


@app.route(routing_table['admin']['dashboard'] + '/<device>/<filename>/check',
           methods=['POST'])
def check_annotate(device, filename):
    confirm = request.json.get('confirm')
    nchamber = request.json.get('nchamber')
    if confirm is None or nchamber is None:
        return "", 400
    _logger.info('Check annotate file: {}/{} with {} chambers as {}'.format(
        device, filename, nchamber, confirm
    ))
    try:
        dicom_service.confirm.set_confirm_on_file(
            device, filename, confirm, nchamber
        )
        return "Success"
    except Exception as e:
        _logger.error(e)
        return "Error", 500


def extract_url_and_filename(data):
    # get json and gif 's url
    for row in data:
        row['url'] = urllib.parse.urljoin(
            request.host_url,
            urllib.parse.quote('data/json_data/{}'.format(row['path']))
        )
    for row in data:
        deviceID = str(row['device'])
        filename = str(row['path'].split('/')[-1].replace('.json', ''))
        gif_url = dicom_service.view.get_gif_url(filename, deviceID)
        row['filename'] = filename
        row['gif_url'] = urllib.parse.urljoin(
            request.host_url, urllib.parse.quote(gif_url)) if gif_url else None
        row['gif_name'] = "{}__{}.gif".format(deviceID, filename)


def check_file_confirmation(data):
    for row in data:
        row['confirm'] = dicom_service.confirm.get_confirm_on_file(
            str(row['device']), str(row['filename'])
        )
    return data
