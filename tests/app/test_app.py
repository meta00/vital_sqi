from dash.testing.application_runners import import_app
from conftest import pytest_setup_options

def test_homepage_app(dash_duo):
    app = import_app('vital_sqi.app.index')
    # dash_duo.driver = ChromeDriverManager().install()
        # Remote(command_executor='http://selenium-hub:4444/wd/hub',
        #                      options=pytest_setup_options())
    dash_duo.start_server(app)
    assert dash_duo.get_logs() == [], "Browser console should contain no error"