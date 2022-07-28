from dash.testing.application_runners import import_app
from conftest import pytest_setup_options
from selenium.webdriver import Chrome,Remote
from webdriver_manager.chrome import ChromeDriverManager
import os

def test_homepage_app(dash_duo):
    app = import_app('vital_sqi.app.index')
    # dash_duo.driver = ChromeDriverManager().install()
    # Chrome('./chromedriver',options=pytest_setup_options())
    # if 'TRAVIS' in os.environ:
    #     dash_duo.driver = ChromeDriverManager().install()
        # dash_duo.driver = ('./chromedriver')
        # dash_duo.driver = \
        #     Remote(command_executor='http://localhost:9222',
        #                          options=pytest_setup_options())
    dash_duo.start_server(app)
    assert dash_duo.get_logs() == [], "Browser console should contain no error"