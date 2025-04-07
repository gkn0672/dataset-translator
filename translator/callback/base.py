class BaseCallback:
    def on_start_init(self, instance):
        """
        Called when the parser has started initializing
        """
        pass

    def on_finish_init(self, instance):
        """
        Called when the parser has finished initializing
        """
        pass

    def on_start_read(self, instance):
        """
        Called when the parser has started reading
        """
        pass

    def on_start_convert(self, instance):
        """
        Called when the parser has started converting
        """
        pass

    def on_finish_convert(self, instance):
        """
        Called when the parser has finished converting
        """
        pass

    def on_start_save_converted(self, instance):
        """
        Called when the parser has started saving
        """
        pass

    def on_finish_save_converted(self, instance):
        """
        Called when the parser has finished saving
        """
        pass

    def on_start_translate(self, instance):
        """
        Called when the parser has started translating
        """
        pass

    def on_finish_translate(self, instance):
        """
        Called when the parser has finished translating
        """
        pass

    def on_error_translate(self, instance, error):
        """
        Called when the parser has encountered an error during translation
        """
        pass

    def on_start_save_translated(self, instance):
        """
        Called when the parser has started saving the translated data
        """
        pass

    def on_finish_save_translated(self, instance):
        """
        Called when the parser has finished saving the translated data
        """
        pass
