from datetime import datetime
from translator.callback.base import BaseCallback


class LogCaptureCallback(BaseCallback):
    def __init__(self):
        self.logs = []

    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def get_logs(self):
        return "\n".join(self.logs)

    def on_start_init(self, instance):
        self.add_log(f"Parser {instance.parser_name} has started initializing")

    def on_finish_init(self, instance):
        self.add_log(f"Parser {instance.parser_name} has finished initializing")

    def on_start_read(self, instance):
        self.add_log(f"Parser {instance.parser_name} has started reading data")

    def on_start_convert(self, instance):
        self.add_log(f"Parser {instance.parser_name} has started converting data")

    def on_finish_convert(self, instance):
        self.add_log(f"Parser {instance.parser_name} has finished converting data")

    def on_start_save_converted(self, instance):
        self.add_log(f"Parser {instance.parser_name} has started saving converted data")

    def on_finish_save_converted(self, instance):
        self.add_log(
            f"Parser {instance.parser_name} has finished saving converted data"
        )

    def on_start_translate(self, instance):
        self.add_log(f"Parser {instance.parser_name} has started translating data")

    def on_finish_translate(self, instance):
        self.add_log(f"Parser {instance.parser_name} has finished translating data")

    def on_error_translate(self, instance, error):
        self.add_log(
            f"Parser {instance.parser_name} encountered an error during translation: {error}"
        )

    def on_start_save_translated(self, instance):
        self.add_log(
            f"Parser {instance.parser_name} has started saving translated data"
        )

    def on_finish_save_translated(self, instance):
        self.add_log(
            f"Parser {instance.parser_name} has finished saving translated data"
        )
