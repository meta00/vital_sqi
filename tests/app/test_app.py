from dash.testing.application_runners import import_app
from vital_sqi.app.index import display_page
from vital_sqi.app.views.dashboard1 import on_data_set_table as load_data_dashboard_1
from vital_sqi.app.views.dashboard2 import on_data_set_table as load_data_dashboard_2
from vital_sqi.app.views import dashboard2
import pandas as pd
from selenium.webdriver.common.by import By


def test_onload_app(dash_duo):
    app = import_app('vital_sqi.app.index')
    dash_duo.start_server(app)
    assert dash_duo.get_logs() == [], "Browser console should contain no error"
    dashboard1_content = display_page('/views/dashboard1')
    assert dashboard1_content.children is not None
    dashboard2_content = display_page('/views/dashboard2')
    assert dashboard2_content.children is not None
    dashboard3_content = display_page('/views/dashboard3')
    assert dashboard3_content.children is not None


def test_dashboard_1(dash_duo):
    mock_data = pd.read_csv('tests/test_data/mock_sqis.csv')
    mock_content_db1 = load_data_dashboard_1(mock_data)
    assert mock_content_db1 is not None


def test_dashboard_2(dash_duo):
    mock_app = import_app('vital_sqi.app.views.dashboard2')
    mock_data = pd.read_csv('tests/test_data/mock_sqis.csv')
    mock_data_rule = {
        "skewness_sqi": {
            "name": "skewness_sqi",
            "def": [
                {"op": ">=", "value": "10", "label": "reject"},
                {"op": ">=", "value": "3", "label": "accept"},
                {"op": "<", "value": "3", "label": "reject"}],
            "desc": "",
            "ref": ""
        }
    }
    mock_content_db2 = load_data_dashboard_2(mock_data, mock_data_rule)
    mock_app.layout = dashboard2.layout
    mock_app.layout.children[2] = mock_content_db2[0]

    dash_duo.start_server(mock_app)
    driver = dash_duo.driver
    driver.get(dash_duo.server_url)

    driver.find_element(
            by=By.ID,
            value='confirm-rule-button')

    assert dash_duo.get_logs() == [], "Confirm Button doesnot load"


#
#
# def test_upload_data(dash_duo):
#     homepage_content = display_page('/homepage')
#     mock_app = dash.Dash(__name__)
#     mock_app.layout = html.Div(id='dashboard-content',children=[
#         dcc.Store(id="dataframe", storage_type='local'),
#         homepage_content
#     ])
#     dash_duo.start_server(mock_app)
#     driver = dash_duo.driver
#     mock_filename = 'mock_sqis.csv'
#     upload_data_field = \
#                 driver.find_element(
#                 by=By.XPATH,
#                 value='//*[@id="upload-data"]/div/input')
#
#     upload_data_field.\
#         send_keys(os.path.join(os.getcwd(),mock_filename))
#
#     assert True
#
#

# def test_upload_data(dash_duo):
#     app = import_app('vital_sqi.app.index')
#     # home_content = display_page('home_content')
#     # temp_layout = app.layout
#     # temp_layout_children = temp_layout.children
#     # temp_layout_children[5] = home_content
#     # temp_layout.children = temp_layout_children
#     # app.layout = temp_layout
#
#     driver = dash_duo.driver
#     dash_duo.start_server(app)
#
#     home_content = display_page('home_content')
#     app.layout.children[5] = home_content
#
#     driver.get(dash_duo.server_url+"/")
#     dash_duo.wait_for_page(dash_duo.server_url+"/homepage",timeout=100)
#     content_div = driver.find_element(by=By.ID, value='page-content')
#
#     print('======== start home content ============')
#     print(content_div)
#     print(content_div.get_attribute("innerHTML"))
#     # driver.execute_script("arguments[0].innerHTML =" + str(home_content), content_div)
#     # content_div = home_content
#     # print(content_div)
#     # upload_data_field = \
#     #         driver.find_element(
#     #             by=By.XPATH,
#     #             value='//*[@id="upload-data"]/div/input')
#     # mock_filename = os.path.join(os.getcwd(), 'mock_sqis.csv')
#     # upload_data_field.send_keys(mock_filename)
#     print('=========end home content ===========')



# def test_upload_data(dash_duo):
#     app = import_app("vital_sqi.app.index")
#     dash_duo.start_server(app)
#     driver = dash_duo.driver
#     driver.get(dash_duo.server_url)
#     print('====================')
#     print(dash_duo.server_url)
#     WebDriverWait(driver).until(lambda x:
#                                 x.find_element(
#                                     by=By.XPATH,
#                                     value='//*[@id="upload-data"]/div/input'))
#     mock_filename = os.path.join(os.getcwd(), 'mock_sqis.csv')
#     upload_data_field = \
#         driver.find_element(
#             by=By.XPATH,
#             value='//*[@id="upload-data"]/div/input')
#     upload_data_field.send_keys("../test_data/mock_sqis.csv")
#     assert True


# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
#
# driver = Chrome()
#
# dc = DesiredCapabilities.CHROME
# driver.capabilities['loggingPrefs'] = { 'browser':'ALL' }
# driver.caps['goog:loggingPrefs'] = { 'browser':'ALL' }
#
# # mock_app = import_app('vital_sqi.app.views.dashboard2')
#
# driver.get('http://127.0.0.1:8050/')
# driver.start_client()
# mock_filename = 'mock_sqis.csv'
# upload_data_field = \
#                 driver.find_element(
#                 by=By.ID,
#                 value='confirm-rule-button')
# upload_data_field.send_keys(os.path.join(os.getcwd(),'mock_sqis.csv'))
#
# print(driver.get_log('browser'))
# print(driver.get_log('driver'))

# upload_data_field = \
#                 driver.find_element(
#                 by=By.XPATH,
#                 value='//*[@id="dashboard-content"]/store')
# sleep(4)
# print('donnneeee')
# homepage_content = display_page('/homepagez')
# mock_app = dash.Dash(__name__)
# mock_app.layout = homepage_content
# mock_app.run_server(debug=True)
#
# driver = webdriver.Chrome()
# driver.get('http://127.0.0.1:8050/')
# upload_data_field = \
#                 driver.find_element(
#                 by=By.XPATH,
#                 value='//*[@id="upload-data"]/div/input')
# mock_filename = 'mock_sqis.csv'
# driver = webdriver.Chrome()
# driver.get('http://127.0.0.1:8050/')
# app = import_app('vital_sqi.app.index')
# home_content = display_page('')
# content_div = driver.find_element(by=By.ID,value='page-content')
#
# content_div = home_content
# upload_data_field = driver.find_element(by=By.XPATH,
#                     value='//*[@id="upload-data"]/div/input')
# mock_filename = os.path.join(os.getcwd(),'mock_sqis.csv')
# upload_data_field.send_keys(mock_filename)
#
# print('ahihi')
# server = app.server
# chrome.get(app)
# print(chrome)