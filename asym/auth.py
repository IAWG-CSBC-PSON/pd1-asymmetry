import asym
from tornado.web import RequestHandler

try:
    prefix = "/" + asym.url_prefix
except AttributeError:
    prefix = ""

login_url = "/login"

LOGIN_HTML = fr"""
<html>
<body>
<form action="{prefix}/login" method="post">
Token: <input type="text" name="token">
<input type="submit" value="Sign in">
</form>
</body>
</html>
"""

try:
    with open("valid_tokens", "r") as f:
        valid_tokens = {x.strip() for x in f.readlines()}
except FileNotFoundError:
    valid_tokens = set()


class LoginHandler(RequestHandler):
    def get(self):
        self.write(LOGIN_HTML)

    def get_current_user(self):
        return self.get_secure_cookie("user")

    def post(self):
        token = self.get_argument("token")
        if token in valid_tokens:
            self.set_secure_cookie("user", token)
            self.redirect(prefix + "/")
            return
        self.redirect(prefix + "/login")


def get_user(request_handler):
    user = request_handler.get_secure_cookie("user")
    if not user:
        request_handler.redirect(prefix + "/login")
    return user
