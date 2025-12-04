class UserBasedCF:
    def __init__(self, R, neighbors, user_means, item_means, global_mean):
        self.R = R
        self.neighbors = neighbors
        self.user_means = user_means
        self.item_means = item_means
        self.global_mean = global_mean

    def predict(self, user_id, movie_id):
        if movie_id not in self.R.columns:
            return self.global_mean

        bu = self.user_means[user_id] - self.global_mean
        bi = self.item_means[movie_id] - self.global_mean
        baseline = self.global_mean + bu + bi

        num = 0.0
        den = 0.0

        for nid, sim in self.neighbors[user_id].items():
            r_nv = self.R.loc[nid, movie_id]
            if not np.isnan(r_nv):
                bu_v = self.user_means[nid] - self.global_mean
                baseline_nv = self.global_mean + bu_v + bi

                num += sim * (r_nv - baseline_nv)
                den += abs(sim)

        return baseline if den == 0 else baseline + num / den
