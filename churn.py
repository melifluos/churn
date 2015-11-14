from __future__ import division
import pandas as pd
import numpy as np
import glob  # read files in dir
from sklearn.preprocessing import StandardScaler  # set all variables to sigma = 1, mu = 0
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
import time
from dateutil.parser import parse
from datetime import datetime

__author__ = 'benchamberlain'


def ml():
    start = time.time()
    customers = pd.read_csv('local_resources/customer/000000_0', sep='\t')
    cust_columns = ['id', 'churn', 'gender', 'country', 'created_on', 'yob', 'premier']
    customers.columns = cust_columns
    customers.set_index('id', inplace=True)
    customers['churn'] -= 1

    print 'read customer table in ', time.time() - start, 's'
    # sample some data
    rows = np.random.choice(customers.index.values, 10000)
    customers = customers.ix[rows]

    #  add total number of purchases and total purchase value features
    receipts_data = process_receipts(customers.index.values)
    customers = customers.join(receipts_data, how='left')

    print 'added receipts in ', time.time() - start, 's'

    #  add number of returns
    return_data = process_returns(customers.index.values)
    customers = customers.join(return_data, how='left')

    print 'added returns in ', time.time() - start, 's'

    #  add web summaries
    # web_data = process_weblogs()
    # customers = customers.join(web_data, how='left')

    # fill nans generated by customers with no transactions of returns
    customers = customers.fillna(value=0)

    # remove accounts created in 1900
    customers = customers[customers['created_on'] != '1900-01-01T00:00:01.000Z']

    # FILTER FOR JUST GB
    customers = customers[customers['country'] == 'UK']
    # DO SOMETHING WITH DATE
    customers['account_duration'] = customers['created_on'].apply(parse_created_on)
    # map gender to 0,1
    customers['female'] = customers['gender'] == 'F'

    # add derived columns
    #customers['purchase freq'] = customers['']

    # Isolate target data
    y = np.array(customers['churn'])
    # We don't need these columns
    to_drop = ['churn', 'created_on', 'country', 'gender']
    churn_feat_space = customers.drop(to_drop, axis=1)
    print churn_feat_space.describe()

    print "Sample data:"
    print churn_feat_space.head(6)

    # Pull out features for future use
    features = churn_feat_space.columns

    X = churn_feat_space.as_matrix().astype(np.float)
    print y.shape
    print y

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print X

    print "Feature space holds %d observations and %d features" % X.shape
    print "Unique target labels:", np.unique(y)

    print "Support vector machines:"
    print "%.3f" % accuracy(y, run_cv(X, y, SVC))
    print "Random forest:"
    print "%.3f" % accuracy(y, run_cv(X, y, RF))
    print "K-nearest-neighbors:"
    print "%.3f" % accuracy(y, run_cv(X, y, KNN))


CHURN_DIC = {1: 'ACTIVE', 2: 'CHURNED'}
PREMIER_DIC = {1: 'PENDING', 2: 'ACTIVE', 3: 'ACTIVE', 4: 'CANCELLED', 5: 'LAPSED', 6: 'DORMANT'}
DIVISION_DIC = {4: 'MENS_OUTLET', 5: 'MENS', 6: 'WOMENS_OUTLET', 7: 'WOMENS'}
SOURCE_DIC = {1: 'FULL_PRICE', 2: 'DISCOUNT_CODE', 3: 'SALES_PURCHASE', 4: 'OTHER', 10: 'RETURNS'}


def process_returns(ids):
    """
    process the returns table
    :return:
    """
    returns = pd.read_csv('local_resources/returns/000000_0', sep='\t')
    ret_columns = ['id', 'product_id', 'division', 'source', 'qty', 'date', 'receipt', 'return_id', 'return_action',
                   'return_reason']
    returns.columns = ret_columns

    ret_drop_col = ['source', 'qty']  # constant across data
    returns.drop(ret_drop_col, inplace=True, axis=1)


    grouped = returns[['id', 'return_id', 'return_action']].groupby(['id', 'return_action']).count()
    grouped.reset_index(inplace=True)
    return grouped.pivot('id', 'return_action', 'return_id')


def process_receipts(ids):
    """
    process receipts table
    :return:
    """
    rec_columns = ['id', 'product_id', 'division', 'source', 'qty', 'date', 'receipt', 'price']
    receipts = read_dir('local_resources/receipts/0*', rec_columns)
    receipts = receipts[receipts['id'].isin(ids)]
    receipts['delta date'] = receipts['date'].apply(parse_created_on)
    grouped = receipts[['id', 'qty', 'price', 'delta date']].groupby('id').agg({'qty': np.sum, 'price': np.sum, 'delta date': np.min})
    grouped.columns = ['days_since_last_receipt', 'total spend', 'total_items']

    grouped_div = receipts[['id', 'division', 'qty', 'price']].groupby(['id', 'division']).sum()
    grouped_div.reset_index(inplace=True)
    div_qty = grouped_div.pivot('id', 'division', 'qty')
    div_qty.columns = ['div4_qty', 'div5_qty', 'div6_qty', 'div7_qty']
    div_price = grouped_div.pivot('id', 'division', 'price')
    div_price.columns = ['div4_price', 'div5_price', 'div6_price', 'div7_price']


    grouped_source = receipts[['id', 'source', 'qty', 'price']].groupby(['id', 'source']).sum()
    grouped_source.reset_index(inplace=True)
    source_qty = grouped_source.pivot('id', 'source', 'qty')
    source_qty.columns = ['source1_qty', 'source2_qty', 'source3_qty', 'source4_qty']
    source_price = grouped_source.pivot('id', 'source', 'price')
    source_price.columns = ['source1_price', 'source2_price', 'source3_price', 'source4_price']


    #grouped_source = receipts[['id', 'division', 'qty', 'price']].groupby(['id', 'source']).sum()
    return pd.concat([div_qty, div_price, source_qty, source_price], axis=1)


def process_weblogs():
    """
    process the web summary data
    :return:
    """
    web_columns = ['id', 'country', 'start_time', 'site', 'page_view_count', 'event_count', 'user_agent', 'screen_res',
                   'browser_size', 'product_view_count', 'distinct_product_view_count', 'added_to_bag_count',
                   'product_saved_from_product_count',
                   'product_saved_from_category_count', 'distinct_products_purchased', 'total_products_purchased']

    web = read_dir('local_resources/sessionsummary/0*', web_columns)
    drop_columns = ['user_agent', 'screen_res', 'browser_size', 'start_time', 'site', 'country']  # not relevant

    web.drop(drop_columns, axis=1, inplace=True)

    grouped = web.groupby('id').sum()
    return grouped


def parse_created_on(row):
    """
    convert date to a time delta
    :param row: a row of the customer table
    :return: the time delta in days
    """
    row_date = parse(row)
    row_date = row_date.replace(tzinfo=None)
    diff = datetime.now() - row_date
    return diff.days


def read_dir(match_str, col_names):
    """
    Read all data in a folder into a single pandas DataFrame
    :param match_str:
    :return:
    """
    start = time.time()
    print 'matching for: ', match_str
    files = glob.glob(match_str)
    print 'files to run: ', files
    all_data = pd.DataFrame(columns=col_names)
    for count, f in enumerate(files):
        data = pd.read_csv(f, sep='\t')
        data.columns = col_names
        all_data = pd.concat([all_data, data], axis=0)
        print count, ' files read in ', time.time() - start, ' seconds'
        print 'shape of data frame: ', all_data.shape
    return all_data


def read_data():
    """
    Read into pandas DataFrames
    :return:
    """
    customers = pd.read_csv('local_resources/customer/000000_0', sep='\t')
    cust_columns = ['id', 'churn', 'gender', 'country', 'created_on', 'yob', 'premier']
    customers.columns = cust_columns
    customers['churn'] -= 1

    web_columns = ['id', 'country', 'star_time', 'site', 'page_view_count', 'event_count', 'user_agent', 'screen_res',
                   'browser_size', 'product_view_count', 'distinct_product_view_count', 'added_to_bag_count',
                   'product_saved_from_product_count',
                   'product_saved_from_category_count', 'distinct_products_purchased', 'total_products_purchased']

    web = read_dir('local_resources/sessionsummary/0*', web_columns)
    drop_columns = ['user_agent', 'screen_res', 'browser_size']  # not relevant

    web.drop(drop_columns, axis=1, inplace=True)

    print 'web', web.shape
    print web.head()
    print 'receipts', receipts.shape
    print receipts.head()

    return web, receipts


def run_cv(X, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)


if __name__ == '__main__':
    start = time.time()
    ml()
    print 'ran in time', time.time() - start, 's'
