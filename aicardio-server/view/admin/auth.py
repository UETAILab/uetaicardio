import logging

from flask import session, request, flash, redirect, url_for

from config import ADMIN_PASSWORD
from view.admin import app, jinja_env
from view.routes import routing_table

_logger = logging.getLogger(__name__)

admin_dashboard_url = routing_table['admin']['dashboard']


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('logged-in'):
        return redirect(admin_dashboard_url)
    error = None
    if request.method == 'POST':
        password = request.form.get("pass")
        if not password or password not in set(ADMIN_PASSWORD):
            error = "Wrong password!!!"
        else:
            session['logged-in'] = True
            return redirect(admin_dashboard_url)
    login_template = jinja_env.get_template('login.html')
    return login_template.render(error=error)


@app.route('/admin/logout', methods=['GET', 'POST'])
def admin_logout():
    session.clear()
    return redirect(url_for('view.admin_login'))


def authorize(f):
    def decorated_func(*args, **kwargs):
        if session.get('logged-in'):
            return f(*args, **kwargs)
        else:
            return redirect(url_for('view.admin_login'))
    return decorated_func
