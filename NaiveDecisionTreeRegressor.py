import numpy as np

np.seterr(all='raise')


class NaiveDecisionTreeRegressor:
    """https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor"""

    def __init__(self, min_samples_split=2, min_samples_leaf=1):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self._fit_init(X, y)
        no_split = False
        while not no_split:
            no_split = True
            new_leaves = []
            for leaf_idx in self._leaves:
                leaf = self._tree[leaf_idx]
                if len(leaf["y"]) >= self.min_samples_split:
                    no_split = False
                    j, s, X1, y1, X2, y2 = self._split_leaf(
                        leaf["X"], leaf["y"])

                    self._tree[leaf_idx] = {"j": j, "s": s}
                    self._append_tree(2 * leaf_idx + 2)
                    self._tree[2 * leaf_idx +
                               1] = {"X": X1, "y": y1, "c": y1.mean()}
                    self._tree[2 * leaf_idx +
                               2] = {"X": X2, "y": y2, "c": y2.mean()}
                    new_leaves.append(2 * leaf_idx + 1)
                    new_leaves.append(2 * leaf_idx + 2)
                else:
                    new_leaves.append(leaf_idx)
            self._leaves = new_leaves
        return self

    def _fit_init(self, X, y):
        self._m = y.shape[0]
        self._d = X.shape[1]
        self._tree = [{"X": X, "y": y, "c": y.mean()}]
        self._leaves = [0]

    def _append_tree(self, n):
        while len(self._tree) - 1 < n:
            self._tree.append({})

    def _split_leaf(self, X, y):
        best_mse = float("inf")
        best_j = 0
        best_s = X[0][0]
        for j in range(self._d):
            for i in range(1, X.shape[0]):
                s = X[i, j]
                X1 = X[((X < s)[:, j])]
                y1 = y[((X < s)[:, j])]
                X2 = X[((X >= s)[:, j])]
                y2 = y[((X >= s)[:, j])]
                if min(y1.shape[0], y2.shape[0]) < self.min_samples_leaf:
                    continue
                mse = self._get_mse(y1, y2)
                if mse < best_mse:
                    best_X1 = X1
                    best_y1 = y1
                    best_X2 = X2
                    best_y2 = y2
                    best_mse = mse
                    best_j = j
                    best_s = s
        return best_j, best_s, best_X1, best_y1, best_X2, best_y2

    def _get_mse(self, y1, y2):
        return np.var(y1) * y1.shape[0] + np.var(y2) * y2.shape[0]

    def predict(self, X):
        yhats = []
        for x in X:
            cur = 0
            while "c" not in self._tree[cur]:
                j = self._tree[cur]["j"]
                s = self._tree[cur]["s"]
                if x[j] < s:
                    cur = 2 * cur + 1
                else:
                    cur = 2 * cur + 2
            yhat = self._tree[cur]["c"]
            yhats.append(yhat)
        return np.array(yhats)

    def score(self, X, y):
        y_true = y
        y_pred = self.predict(X)
        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        return 1 - u / v


if __name__ == '__main__':
    # X = [[0, 0], [2, 2]]
    # y = [0.5, 2.5]
    # clf = DecisionTreeRegressor()
    # clf = clf.fit(X, y)
    # print(clf.predict([[1, 1]]))
    # print(clf._tree)
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print(X_train.shape, X_test.shape)
    naive_clf = NaiveDecisionTreeRegressor(min_samples_split=50)

    naive_clf.fit(X_train, y_train)
    print(naive_clf.score(X_train, y_train))
    print(naive_clf.score(X_test, y_test))
    clf = DecisionTreeRegressor(min_samples_split=50)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    from sklearn import linear_model
    reg = linear_model.Ridge(alpha=.25)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))
