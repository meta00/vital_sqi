from selenium.webdriver.chrome.options import Options

# driver = Remote(command_executor='http://localhost:4444/wd/hub')
def pytest_setup_options():
    options = Options()
    options.add_argument('--disable-gpu')
    return options