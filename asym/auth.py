from tornado.web import RequestHandler

login_url = "/login"

LOGIN_HTML = r"""
<html>
<body>
<form action="/login" method="post">
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
    pass


class LoginHandler(RequestHandler):
    def get(self):
        self.write(LOGIN_HTML)

    def get_current_user(self):
        return self.get_secure_cookie("user")

    def post(self):
        token = self.get_argument("token")
        if token in valid_tokens:
            self.set_secure_cookie("user", token)
            self.redirect("/")
            return
        self.redirect("/login")


def get_user(request_handler):
    return request_handler.get_secure_cookie("user")
