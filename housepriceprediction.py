import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('housing.csv')
df.head()

df.info()

print(df.describe())

print(df['ocean_proximity'].value_counts())

print(df.isnull().sum())

#sns.countplot(x=df['ocean_proximity'],data=df)
#plt.ylabel("No. of counts")
#plt.xlabel("Ocean Types")
#plt.show()

#df.hist(figsize=(15,12),bins=50)
#plt.title("Feature Distributions")
#plt.show()

df['income_category'] = pd.cut(df['median_income'],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5])
print(df['income_category'].value_counts())
plt.hist(df['income_category'],bins=20)
#plt.show()

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, stratify= df['income_category'], random_state= 42)

train.drop(columns = 'income_category' , inplace = True)
test.drop(columns = 'income_category' , inplace = True)

train_copy=train.copy()

plt.figure(figsize=(10,7))
sns.scatterplot(data = train_copy,x = 'latitude', y='longitude',hue= 'median_house_value',palette='coolwarm')
#plt.show()

corr_matrix=train_copy.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True)
#plt.show()

print(corr_matrix['median_house_value'].sort_values(ascending=False))


copy_train=train.copy()
X_train=copy_train.drop(columns='median_house_value')
y_train=copy_train['median_house_value'].copy()

from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True, add_rooms_per_household=True, add_population_per_household=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.add_rooms_per_household = add_rooms_per_household
        self.add_population_per_household = add_population_per_household
    def fit(self, X, y=None):
        return self
    def transform(self,X, y=None):
        if self.add_bedrooms_per_room==1:
            bedrooms_per_room=X[:, bedrooms_ix] / X[:, rooms_ix]
            bedrooms_per_room=np.log(bedrooms_per_room+1)
            X=np.c_[X, bedrooms_per_room]
        if self.add_rooms_per_household==1:
            rooms_per_household=X[:, rooms_ix] / X[:, households_ix]
            rooms_per_household=np.log(rooms_per_household+1)
            X=np.c_[X, rooms_per_household]
        if self.add_population_per_household==1:
            population_per_household=X[:, population_ix] / X[:, households_ix]
            population_per_household=np.log(population_per_household+1)
            X=np.c_[X, population_per_household]
            X[rooms_ix]=np.log(X[rooms_ix]+1)
            X[bedrooms_ix]=np.log(X[bedrooms_ix]+1)
            X[population_ix]=np.log(X[population_ix]+1)
            X[households_ix]=np.log(X[households_ix]+1)
        return X
    
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline= Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attrib_adder', CombinedAttributeAdder(add_bedrooms_per_room=True, add_rooms_per_household=True, add_population_per_household=False)),
    ('std_scaler', StandardScaler()),
])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
num_attribs = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_attribs=['ocean_proximity']
full_pipline=ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])
processed_X_train=full_pipline.fit_transform(X_train)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(processed_X_train, y_train)


predictions=reg.predict(processed_X_train)

from sklearn.metrics import mean_squared_error, r2_score
error=mean_squared_error(y_train, predictions)
error=np.sqrt(error)
print(error)


from sklearn.tree import DecisionTreeRegressor

tree_reg=DecisionTreeRegressor()

tree_reg.fit(processed_X_train, y_train)

tree_predictions=tree_reg.predict(processed_X_train)
tree_error=mean_squared_error(y_train, tree_predictions)
tree_error

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, processed_X_train, y_train, scoring='neg_mean_squared_error', cv=10)
tree_rsme_scores=np.sqrt(-scores)
tree_rsme_scores.mean()

lin_reg_scores=cross_val_score(reg, processed_X_train, y_train, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores=np.sqrt(-lin_reg_scores)
lin_rmse_scores.mean()

from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor()
forest_reg.fit(processed_X_train, y_train)
forest_reg_scores=cross_val_score(forest_reg, processed_X_train, y_train, scoring='neg_mean_squared_error', cv=10)
forest_rmse=np.sqrt(-forest_reg_scores.mean())
forest_rmse

from sklearn.model_selection import GridSearchCV
param_grid=[
    {'n_estimators': [3, 10, 30], 'max_features':[2, 4, 6, 8]},
    {'bootstrap':[False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]}
]
grid_search=GridSearchCV(
    forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True
)
grid_search.fit(processed_X_train, y_train)


grid_search.best_params_

grid_search.best_estimator_

results = grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


final_model=grid_search.best_estimator_
X_test=test.drop(columns='median_house_value')
y_test=test['median_house_value']
processed_X_test=full_pipline.transform(X_test)
final_predictions=final_model.predict(processed_X_test)
final_mse=mean_squared_error(y_test, final_predictions)
final_mse=np.sqrt(final_mse)
print(final_mse)