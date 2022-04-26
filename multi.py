import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection._split import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

# 2 preprossing

# 2.1 load dataset from csv
dataset = pd.read_csv("dataset.csv")

# 2.2 missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
Yas = dataset.iloc[:, 1:4].values

# test
print(Yas)

imputer = imputer.fit(Yas[:, 1:4])
Yas[:, 1:4] = imputer.transform(Yas[:, 1:4])

# test
print(Yas)

Ulke = dataset.iloc[:, 0:1].values

# test
print(Ulke.reshape((22, 1)))

# 2.2 encoder
le = preprocessing.LabelEncoder()
one_hot = preprocessing.OneHotEncoder()

Ulke[:, 0] = le.fit_transform(Ulke[:, 0])
Ulke_one_hot = one_hot.fit_transform(Ulke).toarray()

c = dataset.iloc[:, -1:].values
c[:, -1] = le.fit_transform(c[:, -1])
c = one_hot.fit_transform(c).toarray()

# test
print(Ulke_one_hot)

# 2.3 nump to dataframe
result = pd.DataFrame(data=Ulke_one_hot, index=range(22),
                      columns=["fr", "tr", "us"])

# test
print(result)

result_2 = pd.DataFrame(data=Yas, index=range(22),
                        columns=["boy", "kilo", "yaş"])


result_3 = pd.DataFrame(data=c[:, 0:1], index=range(22), columns=["cinsiyet"])

# 2.4 combining

concat = pd.concat([result, result_2], sort=False, axis=1)
concat_all = pd.concat([result, result_2, result_3], sort=False, axis=1)
print(concat_all)

# 2.5 split values
# find cinsiyet
x_train, x_test, y_train, y_test = train_test_split(
    concat, result_3, random_state=0, test_size=0.33)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# 2.6 scale
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print(y_pred)
print(y_test)


# find boy
boy = concat_all.iloc[:, 3:4].values
print(boy)

left = concat_all.iloc[:, :3]
right = concat_all.iloc[:, 4:]

print(left)
print(right)

concat_2 = pd.concat([left, right], sort=False, axis=1)
print(concat_2)

x_train, x_test, y_train, y_test = train_test_split(
    concat_2, boy, random_state=0, test_size=0.33)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
score = regressor.score(x_test, y_test)

print(y_pred)
print(y_test)
print("score: {0}".format(score))

# backward elimination
X = np.append(arr=np.ones((22, 1)).astype(int), values=concat_2, axis=1)

print(X)

X_l = concat_2.iloc[:, [0, 1, 2, 3, 4, 5]].values

print(X_l)

X_l = np.array(X_l, dtype=float)
# boy ile dizi arasında bağlantı kurar
model = sm.OLS(boy, X_l).fit()
print(model.summary())

#4. elemanın P değeri yüksek elnmesi gerek
X_l = concat_2.iloc[:, [0, 1, 2, 3, 5]].values
X_l = np.array(X_l, dtype=float)
# boy ile dizi arasında bağlantı kurar
model = sm.OLS(boy, X_l).fit()
print(model.summary())