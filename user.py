from werkzeug.security import generate_password_hash, check_password_hash 

class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = generate_password_hash(password)  # Hash the password for security

    def check_password(self, password):
        return check_password_hash(self.password, password)