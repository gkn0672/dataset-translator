class HuggingFaceCallback:
    def on_finish_save_converted(self, instance):
        """
        Called when the parser has finished saving
        """
        # TODO: push the converted data to huggingface hub
        pass
