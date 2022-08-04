from selenium.webdriver.chrome.options import Options
import pytest

# driver = Remote(command_executor='http://localhost:4444/wd/hub')
def pytest_setup_options():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')
    return options


@pytest.fixture
def unwrap():
    def unwrapper(func):
        if not hasattr(func, '__wrapped__'):
            return func

        return unwrapper(func.__wrapped__)

    yield unwrapper