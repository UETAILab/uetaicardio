class HandledError(Exception):
    def __init__(self, error_code, message):
        self.error_code = error_code
        self.message = message

    def __str__(self):
        return 'Error code [%s]: %s' % (self.error_code, self.message)

    def __repr__(self):
        return 'Error code [%s]: %s' % (self.error_code, self.message)
