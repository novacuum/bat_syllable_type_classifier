from datetime import datetime, timedelta
import pandas as pd
from engine.settings import BIRDVOICE_FOLDER


class Bat:
    def __init__(self, id, bubble_start, bubble_end):
        self.id = id
        self.bubble_start = bubble_start
        self.bubble_end = bubble_end

    def to_bubble_days(self, date):
        if date < self.bubble_start or date > self.bubble_end:
            raise Exception('date is not in bubble range')

        return date - self.bubble_start

    def get_half_time(self):
        return self.bubble_start + ((self.bubble_end - self.bubble_start) / 2)


class BatService:
    def __init__(self):
        self.bats = dict()
        self.load()

    def load(self):
        df = pd.read_excel(BIRDVOICE_FOLDER / 'batinfo.xlsx')
        for index, row in df.iterrows():
            bat = Bat(row['id'], row['bubble_start'], row['bubble_start'] + timedelta(days=row['bubble_time']))
            self.bats[bat.id] = bat

    def get_by_id(self, id):
        return self.bats[id]

    def to_bubble_days(self, id, date):
        return self.get_by_id(id).to_bubble_days(date)


