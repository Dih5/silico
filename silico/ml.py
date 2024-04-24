"""Common sklearn predictors"""

# Classification methods
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Regression methods
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

from .common import set_kwargs

# Mapping of user-friendly names to Classifier
classifiers = {"baseline": set_kwargs(DummyClassifier, {"strategy": "stratified"}),
               "logistic": set_kwargs(LogisticRegression, {"solver": "lbfgs", "multi_class": "auto"}),
               "svm": set_kwargs(SVC, {"gamma": "scale"}),
               "k-neighbors": KNeighborsClassifier,
               "decision-tree": DecisionTreeClassifier,
               "random-forest": set_kwargs(RandomForestClassifier, {"n_estimators": 100}),
               "extra-trees": set_kwargs(ExtraTreesClassifier, {"n_estimators": 100}),
               "gradient-boosting": GradientBoostingClassifier,
               "mlp": MLPClassifier, }

# Mapping of user-friendly names to Regressor
regressors = {"baseline": DummyRegressor,
              "linear": LinearRegression,
              "svm": set_kwargs(SVR, {"gamma": "scale"}),
              "k-neighbors": KNeighborsRegressor,
              "decision-tree": DecisionTreeRegressor,
              "random-forest": set_kwargs(RandomForestRegressor, {"n_estimators": 100}),
              "extra-trees": set_kwargs(ExtraTreesRegressor, {"n_estimators": 100}),
              "gradient-boosting": GradientBoostingRegressor,
              "mlp": MLPRegressor, }
