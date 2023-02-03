import re
from operator import itemgetter

from plot.settings import MODEL_COLORS


class Mapper:
    def forward(self, value):
        return value

    def reverse(self, value, *args):
        return value


class MapperCollection(Mapper):
    def __init__(self, *mappers):
        self.mappers = mappers

    def forward(self, value):
        for mapper in self.mappers:
            value = mapper.forward(value)
        return value

    def reverse(self, value, *args):
        for mapper in reversed(self.mappers):
            value = mapper.reverse(value)
        return value


class FunctionMapper(Mapper):
    def __init__(self, function):
        self.mapping = dict()
        self.function = function

    def forward(self, value):
        result = self.function(value)
        if result not in self.mapping:
            self.mapping[result] = value
        return result

    def reverse(self, value, *args):
        return self.mapping[value]


class String2IntMapper(Mapper):
    def __init__(self, start=0):
        self.mapping = list()
        self.start = start

    def forward(self, value):
        if value not in self.mapping:
            self.mapping.append(value)
        return self.start + self.mapping.index(value)

    def reverse(self, value, *args):
        return self.mapping[int(value) - self.start]


class DictMapper(Mapper):
    def __init__(self, mapping: dict):
        self.mapping = mapping
        self.reverse_mapping = {str(value): key for key, value in self.mapping.items()}

    def forward(self, value):
        return self.mapping[value]

    def reverse(self, value, *args):
        return self.reverse_mapping[str(value)]


class Getter:
    def __init__(self, mapper: Mapper):
        self.mapper = mapper

    def get(self, obj):
        return self.mapper.forward(self.getitem_from_object(obj))

    def getitem_from_object(self, obj):
        pass

    def __call__(self, obj):
        return self.get(obj)


class MatchItemValue(Getter):
    def __init__(self, name, pattern, pos, mapper: Mapper):
        super().__init__(mapper)
        self.itemgetter = itemgetter(name)
        self.pattern = pattern
        self.pos = pos

    def getitem_from_object(self, obj):
        match = self.pattern.match(self.itemgetter(obj))
        if isinstance(self.pos, list):
            return [match[index] for index in self.pos]
        return match[self.pos]


class ResultTaskPropGetter(Getter):
    def __init__(self, task_name, prop_name, empty_value, mapper: Mapper = None):
        super().__init__(Mapper() if mapper is None else mapper)
        self.task_name = task_name
        self.prop_name = prop_name
        self.empty_value = empty_value

    def getitem_from_object(self, obj):
        tasks = list(filter(lambda task: task['task'] == self.task_name and self.prop_name in task['props'], obj['pipeline']))
        return self.empty_value if len(tasks) == 0 else tasks[0]['props'][self.prop_name]


class ModelMatcher(MatchItemValue):
    def __init__(self, mapper=None):
        super().__init__('id', re.compile(r'^nn_(.*)_sc[ts]_'), 1, DictMapper(MODEL_COLORS) if mapper is None else mapper)

    def getitem_from_object(self, obj):
        value = super().getitem_from_object(obj)
        if value == 'densNet': return 'denseNet'
        return value


class FeatureMatcher(MatchItemValue):
    def __init__(self):
        super().__init__(
            'id'
            , re.compile(r'.*_(hog|raw)_\d+_?([^/_]*)?')
            , [1, 2]
            , FunctionMapper(lambda v: 'o' if v == 'raw' else '+' if v == 'hog' else '2')
        )

    def getitem_from_object(self, obj):
        return ' '.join(filter(len, super().getitem_from_object(obj)))


