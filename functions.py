from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
import numpy as np # linear algebra
from pandas import read_csv, DataFrame, concat, to_numeric
import pandas as pd
import matplotlib.pyplot as pl

def col_convert(df):
    columns_convert = df.select_dtypes(include=[object]).keys()
    df[columns_convert]= df[columns_convert].apply(lambda x: to_numeric(x.astype(str).str.replace(',', '.'), errors='ignore'))
    return df

def replace_mfloor(x,floor):
    if (np.isnan(x)) or( x == 0)or( x == 1):
        return floor
    else:
        return x

def replace_floor(x,mfloor):
    if x > mfloor:
        if 10 < x < 100:
            return int(x / 10)
        elif x > 100:
            return int(x / 100)
        else:
            return mfloor
    else:
        return x

def replace_build(x):
    if (np.isnan(x)) or( x == 0):
        return -99
    elif x == 20052009:
        return 2007
    elif x == 4965:
        return 1965
    elif x <= 20:
        return 2000 + x
    elif x < 100:
        return 1900 + x
    elif x < 220:
        return 1800 + x
    else:
        return x

def replace_with_median(x,median):
    if (np.isnan(x)) or( x == 0)or( x == 1):
        return median
    else:
        return x

def fix_state(x):
    if x ==33:
        return 3
    else:
        return x

def fill_median(df1,df2,numerical):
    for f in numerical:
        values = list(df1[f].values) + list(df2[f].values)
        median = np.nanmedian(values)
        #print(f,median)
        df1[f] = df1[f].fillna(median)
        df1[f] = df1[f].replace(0, median)
        df2[f] = df2[f].fillna(median)
        df2[f] = df2[f].replace(0, median)
    return df1,df2

def fix_macro(df):
    df = col_convert(df)
    df = df.replace('#!', -99)
    df['child_on_acc_pre_school'] = df['child_on_acc_pre_school'].replace(',', '').astype(float)
    df['modern_education_share'] = df['modern_education_share'].astype(float)
    df['old_education_build_share'] = df['old_education_build_share'].astype(float)
    df = df.drop('timestamp', axis=1)
    df = df.fillna(0)
    assert isinstance(df, object)
    return df

def columns_drop(drop):
    if drop == 'macro':
        col = read_csv("macro_dropped.csv")
    else:
        col = read_csv("columns_dropped.csv")
    list_col_drop = list(col['features'].values)
    return list_col_drop

def fix_col (df):
    full_sq_avg = np.nanmedian(df['full_sq'])
    life_sq_avg = np.nanmedian(df['life_sq'])
    kitch_sq_avg = np.nanmedian(df['kitch_sq'])

    df['full_sq']= df.apply(lambda row: replace_with_median(row['full_sq'],full_sq_avg), axis=1)
    df['life_sq']= df.apply(lambda row: replace_with_median(row['life_sq'],life_sq_avg), axis=1)
    df['kitch_sq']= df.apply(lambda row: replace_with_median(row['kitch_sq'],kitch_sq_avg), axis=1)
    df['sub_area']= df['sub_area'].str.replace(" ","").str.replace("\'","").str.replace("-","")
    df['logruboil'] = np.log( df.oil_urals * df.usdrub )
    df[['floor']] = df[['floor']].fillna(value=0)
    df['max_floor']= df.apply(lambda row: replace_mfloor(row['max_floor'],row['floor']), axis=1)
    df['floor']= df.apply(lambda row: replace_floor(row['floor'],row['max_floor']), axis=1)
    df['build_year'] = df.apply(lambda row: replace_build(row['build_year']), axis=1)
    df['build_age'] = 2020 - df['build_year']
    df['build_age'] = df['build_age'].replace(2119,-99)
    df['flor_prop'] =  (df['floor'] + 1) / (df['max_floor'] + 1)
    df['state'] = df.apply(lambda row: fix_state(row['state']), axis=1)
    df["year"] = df["timestamp"].dt.year
    df["yearmonth"] = df["timestamp"].dt.year*100 + df["timestamp"].dt.month
    df["building"] = df['sub_area'] + df['metro_km_avto'].astype(str) + df['public_transport_station_km'].astype(str)
    return df

def test_integrity(features,test_features):
    columns = features.columns
    test_columns = test_features.columns
    for i in range(len(columns)):
        if columns[i] != test_columns[i]:
            print (f'error at training: {columns[i]} testing: {test_columns[i]}')

class RidgeTransformer(Ridge, TransformerMixin):
    def transform(self, X, *_):
        return self.predict(X)


class xgbTransformer(XGBRegressor, TransformerMixin):
    def transform(self, X, *_):
        return self.predict(X)


class KNeighborsTransformer(KNeighborsRegressor, TransformerMixin):
    def transform(self, X, *_):
        return self.predict(X)


def build_model():
    ridge_transformer = Pipeline([
        ('scaler', StandardScaler()),
        ('poly_feats', PolynomialFeatures()),
        ('ridge', Ridge())
    ])

    pred_union = FeatureUnion(
        transformer_list=[
            ('ridge', ridge_transformer),
            ('xgb', XGBRegressor()),
            ('knn', KNeighborsRegressor())
        ],
        n_jobs=-1
    )

    model = Pipeline([
        ('pred_union', pred_union),
        ('lin_regr', LinearRegression())
    ])

    return model


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped = True
        while dropped:
            # Loop repeatedly until we find that all columns within our dataset
            # have a VIF value we're happy with.
            variables = X.columns
            dropped = False
            vif = []
            new_vif = 0
            for var in X.columns:
                new_vif = variance_inflation_factor(X[variables].values, X.columns.get_loc(var))
                vif.append(new_vif)
                if np.isinf(new_vif):
                    break
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print('Dropping {} with vif= {}'.format(X.columns[maxloc], max_vif))
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X

def pca_results(good_data, pca, ind):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = ['Dimension_{}'.format(i+ind) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = pl.subplots(figsize = (12,4))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar', legend = False);
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

    # Return a concatenated DataFrame
    return concat([variance_ratios, components], axis = 1)