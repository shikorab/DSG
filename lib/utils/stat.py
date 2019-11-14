class Stat:

    def __init__(self):
        self.sum = {}
        self.sum_square = {}
        self.count = {}

    def add(self, key, value):
        self.count[key] = self.count.get(key, 0) + 1
        self.sum[key] = self.sum.get(key, 0.0) + value
        self.sum_square[key] = self.sum_square.get(key, 0.0) + value ** 2

    def add_dict(self, dict):
        for key in dict:
            self.add(key, dict[key])

    def print_stat(self):
        for key in self.count:
            name = key
            count = self.count[key]
            sum = self.sum[key]
            square = self.sum_square[key]
            avg = sum / count
            std = (square / count - (avg) ** 2) ** 0.5
            print("%s - count %d avg %.2g std %.2g" % (name, count, avg, std))

    def get_count(self, keys):
        return {key: self.count.get(key, 0) for key in keys}

    def get_avg(self, keys):
        return {key: self.sum.get(key, 0) / self.count.get(key, 0) for key in keys if self.count.get(key, 0) != 0}

    def get_std(self, keys):
        return {key: (self.square.get(key, 0) / self.sum.get(key, 0) - (
                self.sum.get(key, 0) / self.count.get(key, 0)) ** 2) ** 0.5 for key in keys if key in self.count}
