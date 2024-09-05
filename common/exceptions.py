
class ParsingError(Exception):
    def __init__(self, message, code):
        super().__init__()
        self.message=message
        self.code=code

    def __str__(self):
        return self.message