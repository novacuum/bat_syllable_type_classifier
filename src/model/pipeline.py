

class JsonPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def has_task(self, name):
        return self.get_task(name) is not None

    def get_task(self, name):
        for task in filter(lambda action: action['task'] == name, self.pipeline):
            return task
        return None

    def get_dataset_name(self):
        return self.get_task('AudioLoadTask')['props']['dataset_name']
