from math import pow, sqrt
import json


class OutlierModel:
    def __init__(self):
        self.yeeted = 0
        self.stats = {
            "area": [],
            "circularity": [],
            "mean_intensity": [],
            "perimeter": []
        }

    def add_sample(self, features):
        for feature, channels in features.items():
            for channel, v in enumerate(channels):
                # Welford's online algorithm
                stats = self.stats[feature][channel]
                stats["n"] += 1
                old_m = stats["m"]
                stats["m"] = ((stats["n"]-1)*stats["m"] + v) / stats["n"]
                stats["M"] = stats["M"] + (v-old_m)*(v-stats["m"])
                if stats["n"] > 1:
                    stats["s"] = stats["m"]/(stats["n"]-1)

    def write(self):
        with open("outlier_model.json", 'w+') as f:
            json.dump({
                feature: [
                    {stat: v for stat, v in channel.items()}
                    for channel in channels
                ]
                for feature, channels in self.stats.items()
            }, f)

    def read(self, file):
        with open(file, 'r') as f:
            self.stats = json.load(f)

    def is_no_outlier(self, features):
        votes = 0
        for feature in features:
            feature_name = feature[0]
            channels = feature[1]
            for channel, v in enumerate(channels):
                stat = self.stats[feature_name][channel]
                bound = .5 * sqrt(stat["variance"])
                if stat["mean"] - bound < v < stat["mean"] + bound:
                    votes -= 1
                else:
                    votes += 1
        return votes < 0

    # featurs: rdd
    def train(self, features):
        result = features.combineByKey(feature_to_c, features_merge_c, merge_feature_cs).collect()
        print(result)
        for feature in result:
            self.stats[feature[0]] = [{"mean": channel[1], "variance": channel[2]} for channel in feature[1]]
        self.write()


def feature_to_c(feature):
    return [[1, v, 0] for v in feature]


def features_merge_c(combiner, feature):
    # print("\n" * 10)
    # print(combiner)
    # print("\n" * 10)
    # 0: count
    # 1: mean
    # 2: variance
    for channel, stat in enumerate(feature):
        stats = combiner[channel]
        stats[0] += 1
        old_m = stats[1]
        stats[1] = ((stats[0] - 1) * stats[1] + stat) / stats[0]
        if stats[0] > 1:
            ssd0 = stats[2] * (stats[0]-2)   # old squared sum of differences
            ssd1 = ssd0 + (stat-old_m)*(stat-stats[1])  # new squared sum of differences
            stats[2] = ssd1 / (stats[0] - 1)
    return combiner


def merge_feature_cs(c1, c2):
    for i, stats in enumerate(c1):
        n1 = stats[0]
        n2 = c2[i][0]
        m1 = stats[1]
        m2 = c2[i][1]
        ssd1 = stats[2] * (n1-1)
        ssd2 = c2[i][2] * (n2-1)

        stats[0] += n2                                      # pooled n
        stats[1] = (n1 * m1 + n2 * m2) / (stats[0])         # pooled mean
        ssd3 = ssd1 + ssd2 + pow(n1*(m1-stats[1]), 2)+pow(n2*(m2-stats[1]), 2)  # pooled squared sum of differences
        if stats[0] > 1:                                    # pooled variance
            stats[2] = ssd3 / (stats[0]-1)
    return c1
