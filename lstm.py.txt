import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
import gc
from datetime import datetime, timedelta
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
from kaggle.competitions import twosigmanews
from skimage.io import imread
from skimage.transform import resize
from multiprocessing import Pool
import keras
pd.set_option('max_columns', 50)
windowsize = 10

class SequenceGenerator(keras.utils.Sequence):
    def __init__(self, df, num_cols, cat_cols, lstm_cols = [], window=windowsize, batch_size=128, train=True):
        self.batch_size = batch_size
        self.train = train
        self.window = window
        self.data = df
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.lstm_cols = lstm_cols
        self.cols = num_cols + cat_cols
        self.y01 = self.data.returnsOpenNextMktres10.map(lambda x: 0 if x < 0 else 1)
        self.allrows = range(self.data.shape[0])
        self.shuffle = False

    def generate(self):
        X, y, d, r, u = {'num':[], 'lstm':[]}, [], [], [], []
        for cat in cat_cols:
            X[cat] = []
        # Subtract batch_size and window to make sure that I don't cross the boundary
        for seq in range(0, self.data.shape[0] - self.batch_size - self.window, self.batch_size):
            # Todo tabuleado desde aca hasta el def step function
            X['num'] = self.data[self.num_cols].iloc[(seq + self.window - 1):(seq + self.window + self.batch_size - 1)].values
            for subseq in range(seq, seq + X['num'].shape[0]):
                X['lstm'].append(self.data[self.lstm_cols].iloc[subseq:(subseq + self.window)].values)
                #for _ in range(windowsize - X['lstm'][-1].shape[0]): # Repeat values if there's missing history
                #    print('I shouldnt have entered here')
                #    X['lstm'][-1] = X['lstm'][-1].iloc[0:1].append(X['lstm'][-1])
            # The next for loop will enter the LSTM part. I only select members of length of the window
            for cat in cat_cols:
                X[cat] = self.data[cat].iloc[(seq + self.window - 1):(seq + self.window + self.batch_size - 1)].values
            y = self.y01.iloc[(seq + self.window - 1):(seq + self.window + self.batch_size - 1)].values
            #if self.train:
            #    d = self.data.time.iloc[(seq + self.window - 1):(seq + self.window + self.batch_size - 1)].values
            #    r = self.data.returnsOpenNextMktres10.iloc[(seq + self.window - 1):(seq + self.window + self.batch_size - 1)].values
            #    u = self.data.universe.iloc[(seq + self.window - 1):(seq + self.window + self.batch_size - 1)].values

            X_ = {'num':np.array(X['num']), 'lstm':np.array(X['lstm'])}
            for cat in cat_cols:
                X_[cat] = np.array(X[cat])
            y_ = np.array(y) 
            r_, u_, d_ = np.array(r),np.array(u), np.array(d)
            X, y, d, r, u = {'num':[], 'lstm':[]}, [], [], [], []
            for cat in cat_cols:
                X[cat] = []
            if self.train:
                yield X_, y_
            else:
                yield X_, y_, r_, u_, d_
                
    def __generate(self, seq):
        print('entered __generate with seq', seq)
        X, y = {'num':[], 'lstm':[]}, []
        for cat in cat_cols:
            X[cat] = []
        # Subtract batch_size and window to make sure that I don't cross the boundary
        #for seq in range(0, self.data.shape[0] - self.batch_size - self.window, self.batch_size):
        X['num'] = self.data[self.num_cols].iloc[(seq + self.window - 1):(seq + self.window + self.batch_size - 1)].values
        for subseq in range(seq, seq + X['num'].shape[0]):
            X['lstm'].append(self.data[self.lstm_cols].iloc[subseq:(subseq + self.window)].values)
        # The next for loop will enter the LSTM part. I only select members of length of the window
        for cat in cat_cols:
            X[cat] = self.data[cat].iloc[(seq + self.window - 1):(seq + self.window + self.batch_size - 1)].values
        y = self.y01.iloc[(seq + self.window - 1):(seq + self.window + self.batch_size - 1)].values
        
        X_ = {'num':np.array(X['num']), 'lstm':np.array(X['lstm'])}
        for cat in cat_cols:
            X_[cat] = np.array(X[cat])
        y_ = np.array(y) 
        print('exiting __generate with seq', seq)
        gc.collect()
        return X_, y_
                
    def __len__(self):
        num_sequences = self.data.shape[0] - self.window 
        steps = num_sequences//self.batch_size
        print('entered the __len__ and returned', steps)
        return steps
        
    def __getitem__(self, index):
        myindex = index * self.batch_size
        print('entered __geitem__ with index', index, 'and my generated index is', myindex)
        X, y = self.__generate(myindex)
        print('index', index, 'will return X, y with sizes', y.shape, X['num'].shape, X['lstm'].shape)
        return X, y
        
    def on_epoch_end(self):
        print('on_epoch_end')
        if self.shuffle == True:
            print("dadasdasd -----------------------------")
            raise ValueError('Shuffeling not implemented yet')        
            
    def steps(self):
        # get number of steps per epoch
        steps = 0
        num_sequences = self.data.shape[0] - self.window
        steps += num_sequences//self.batch_size
        return steps
    
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()
# Remove data before 2009
start = datetime(2016, 1, 1, 0, 0, 0).date()
market_train = market_train.loc[market_train['time'].dt.date >= start].reset_index(drop=True)
news_train = news_train.loc[news_train['time'].dt.date >= start].reset_index(drop=True)
# Preprocess some news and categorical data, also remove some columns that are not used
def preprocess_news(news_train):
    drop_list = [
        'audiences', 'subjects', 'assetName',
        'headline', 'firstCreated', 'sourceTimestamp',
    ]
    news_train.drop(drop_list, axis=1, inplace=True)
    # Factorize categorical columns
    for col in ['headlineTag', 'provider', 'sourceId']:
        news_train[col], uniques = pd.factorize(news_train[col])
        del uniques
    # Remove {} and '' from assetCodes column
    news_train['assetCodes'] = news_train['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    return news_train

def unstack_asset_codes(news_train):
    codes = []
    indexes = []
    for i, values in news_train['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)]*len(explode)
        indexes.extend(repeat_index)
    index_df = pd.DataFrame({'news_index': indexes, 'assetCode': codes})
    del codes, indexes
    gc.collect()
    return index_df
    
def addWeirdColumns(market_train):
    # Add the bolsa data
    bolsa = []
    for i, values in market_train['assetCode'].iteritems():
        try:
            bolsa.append(values.split(".")[1][0])
        except:
            bolsa.append('U')
    market_train['bolsa'] = bolsa
    market_train['month'] = pd.to_datetime(market_train['time']).dt.month
    market_train['dayofweek'] = pd.to_datetime(market_train['time']).dt.dayofweek
    market_train['weekofmonth'] = np.floor(pd.to_datetime(market_train['time']).dt.day / 7)
    market_train['weekofmonth'] = market_train['weekofmonth'].astype(int)
    market_train['returnsClosePrevRaw1'].fillna(0)
    market_train['bollingerExcess'] = 0.0
    market_train['MACD'] = 0.0
    market_train['reversal'] = 0.0
    mytime = time.time()
    if False:
        companylist = np.unique(market_train['assetCode'])
        market_list = []
        mres = market_train.iloc[0:1]
        for company in companylist:
            market_list.append(market_train[company == market_train['assetCode']])
        pool = Pool(processes = 4)
        #if __name__ == '__main__':
        mres = pd.concat(pool.map(work_on_company, market_list))
        pool.close()
        pool.join()
        print('individual loops took these many seconds', time.time() - mytime)
        market_train['volume'] = market_train['volume'].clip(-1, 1)
    return(market_train)

def work_on_company(m):
    global market_train
    # Calculate price
    # Bollinger Bands
    # calculate Simple Moving Average with 20 days window
    sma = m.close.rolling(window=windowsize).mean()
    # calculate the standar deviation
    rstd = m.close.rolling(window=windowsize).std()
    upper_band = sma + 2 * rstd
    lower_band = sma - 2 * rstd
    bolpos = ((m.close - upper_band)/upper_band).values
    bolpos = np.where(bolpos < 0, 0, bolpos)
    bolneg = ((m.close - lower_band)/lower_band).values
    bolneg = np.where(bolneg > 0, 0, bolneg)
    m['bollingerExcess'] = bolpos + bolneg
    emaslow = m.close.ewm(span=22).mean()
    emafast = m.close.ewm(span=7).mean()
    m['MACD'] = emafast - emaslow
    m['reversal'] = m.returnsClosePrevRaw1.rolling(20).sum()
    m['volume'] = m['volume'].shift(1) / m['volume'] - 1
    #market_train.loc[company == market_train['assetCode'],['volume', 'bollingerExcess', 'MACD', 'reversal']] = m[['volume', 'bollingerExcess', 'MACD', 'reversal']]
    return(m)
            
def merge_news_on_index(news_train, index_df):
    news_train['news_index'] = news_train.index.copy()
    # Merge news on unstacked assets
    news_unstack = index_df.merge(news_train, how='left', on='news_index')
    news_unstack.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    return news_unstack

def group_news(news_frame):
    news_frame['date'] = news_frame.time.dt.date  # Add date column
    aggregations = ['mean']
    aggregations = {'urgency':'min', 'takeSequence':'min', 'provider':'max', 'bodySize':'mean', 'companyCount':'max', 
        'headlineTag':'count', 'marketCommentary':'max', 'sentenceCount':'max', 'wordCount':'sum', 'firstMentionSentence':'min',
        'relevance':'max', 'sentimentClass':'max', 'sentimentNegative':'mean', 'sentimentNeutral':'mean', 'sentimentPositive':'mean',
        'sentimentWordCount':'sum', 'noveltyCount12H':'max', 'noveltyCount24H':'max', 'noveltyCount3D':'max', 
        'noveltyCount5D':'max', 'noveltyCount7D':'max', 'volumeCounts12H':'max', 'volumeCounts24H':'max', 'volumeCounts3D':'max', 
        'volumeCounts5D':'max', 'volumeCounts7D':'max', 'countAssetCodes':'sum'}
    # MULTIPLY RELEVANCE TIMES SENTIMENT
    gp = news_frame.groupby(['assetCode', 'date']).agg(aggregations)
    #gp.columns = pd.Index(["{}_{}".format(e[0], e[1]) for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    # Set datatype to float32
    float_cols = {c: 'float32' for c in gp.columns if c not in ['assetCode', 'date']}
    return gp.astype(float_cols)
    
def finalTransformations(market_train):
    market_train['feelingRatio'] = market_train['sentimentWordCount'] / market_train['wordCount']
    market_train.loc[0 == market_train['firstMentionSentence'],'firstMentionSentence'] = market_train[0 == market_train['firstMentionSentence']]['sentenceCount']
    market_train['importanceRatio'] = (market_train['sentenceCount'] - market_train['firstMentionSentence'])/market_train['sentenceCount']
    market_train['noveltyTechnicalA'] = market_train['noveltyCount5D'] - market_train['noveltyCount24H']
    market_train['noveltyTechnicalB'] = market_train['noveltyCount7D'] - market_train['noveltyCount3D']
    market_train['positiveMnegative'] = market_train['sentimentPositive'] - market_train['sentimentNegative']
    market_train['passion'] = market_train['positiveMnegative'] * market_train['relevance']
    market_train['marketCommentary'] = market_train['marketCommentary'] + 1.0
    market_train['provider'] = market_train['provider'] + 1.0
    return(market_train)
    
def merge_market_news(market_train, news_train):
    print('Adding the weird columns')
    market_train = addWeirdColumns(market_train)
    print('preprocessing news...')
    news_train = preprocess_news(news_train)
    print('Done')
    # Let's add some magic to the data count the number of assetcodes in the news bef. I destroy this data below
    news_train['countAssetCodes'] = [i.count('.') for i in news_train['assetCodes']]
    print('Unstack the news...')
    # Now I'm going to unstack the news
    index_df = unstack_asset_codes(news_train)
    print('Done')
    # and merge the news on this frame
    print('Merge the news on this frame...')
    news_unstack = merge_news_on_index(news_train, index_df)
    del news_train, index_df
    gc.collect()
    print('Done')
    # Group by date and asset using simple mean
    print('Group news by date and asset using a simple mean (Think this better there are several news per day per asset)...')
    news_agg = group_news(news_unstack)
    del news_unstack; gc.collect()
    print('Done')
    # Finally, merge on assetCode and Date
    print('Merge both datasets based on asset code and date...')
    market_train['date'] = market_train.time.dt.date
    market_train = market_train.merge(news_agg, how='left', on=['assetCode', 'date'])
    del news_agg
    gc.collect()
    print('Done')
    print('Final Transformations')
    market_train = finalTransformations(market_train)
    market_train.drop(drop_cols, axis=1, inplace=True)
    print('Done')
    return(market_train)
    
def appendNewDays(market, newmarket = []):
    try:
        market.drop(['returnsOpenNextMktres10', 'universe'], axis = 1, inplace = True)
    except:
        pass
    market = market[market.time.isin(np.sort(np.unique(market.time))[-windowsize+1:])]
    try:
        market = market.append(newmarket)
    except:
        pass
    return(market)

cat_cols = ['assetCode', 'bolsa', 'month', 'dayofweek', 'weekofmonth', 'urgency', 'provider']#, 'provider', 'urgency', 'marketCommentary', 'firstMentionSentence', 'sentimentClass']
#cat_cols = ['assetCode']
lstm_cols = ['returnsOpenPrevMktres1', 'relevance', 'volume', 'feelingRatio', 'importanceRatio', 'noveltyTechnicalA', 'noveltyTechnicalB', 'passion', 'positiveMnegative']#, 'MACD', 'reversal']
num_cols = ['close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsOpenPrevMktres10', 'feelingRatio', 'passion']#, 'bollingerExcess']
news_num_cols = ['companyCount', 'sentimentNegative',
                 'sentimentNeutral','sentimentPositive', 'noveltyCount12H',
                 'noveltyCount24H','noveltyCount3D','noveltyCount5D','noveltyCount7D','volumeCounts12H',
                 'volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D', 'countAssetCodes']
num_cols += [word for word in news_num_cols]
all_cols = lstm_cols + num_cols
existing_cols = all_cols + cat_cols + ['returnsOpenNextMktres10', 'universe', 'time']
drop_cols = [i for i in market_train.columns.values if i not in existing_cols]
market_train = merge_market_news(market_train, news_train)
###########################################################################################################################

from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(market_train.index.values, test_size=0.2, random_state=12, shuffle = False)

def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{} for cat in cat_cols]
for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(market_train.loc[train_indices, cat].astype(str).unique())}
    market_train[cat] = market_train[cat].astype(str).apply(lambda x: encode(encoders[i], x))
print('Done')
embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets

from sklearn.preprocessing import StandardScaler
 
#market_train[num_cols] = market_train[num_cols].fillna(market_train[num_cols].mean())
market_train[all_cols] = market_train[all_cols].fillna(0)
print('scaling numerical columns')

scaler = StandardScaler()
market_train[num_cols] = scaler.fit_transform(market_train[num_cols])
market_train[lstm_cols] = scaler.fit_transform(market_train[lstm_cols])
print('Done')

print('Defining the architecture...')
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, LSTM
from keras.losses import binary_crossentropy, mse

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

#categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
categorical_logits = Flatten()(categorical_embeddings[0])
#categorical_logits = Dense(98,activation='relu')(categorical_logits)
#categorical_logits = Dense(64,activation='relu')(categorical_logits)
categorical_logits = Dense(32,activation='relu')(categorical_logits)

numerical_inputs = Input(shape=(len(num_cols),), name='num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)
numerical_logits = Dense(128,activation='relu')(numerical_logits)
numerical_logits = Dense(85,activation='relu')(numerical_logits)

# LSTM part
lstm_inputs = Input(shape = (10, len(lstm_cols)), name='lstm')
lstm_logits = LSTM(128,return_sequences=True)(lstm_inputs)
#lstm_logits = LSTM(32,return_sequences=True)(lstm_logits)
lstm_logits = LSTM(64)(lstm_logits)

logits = Concatenate()([numerical_logits,categorical_logits,lstm_logits])
logits = Dense(64,activation='relu')(logits)
out = Dense(1, activation='sigmoid')(logits)

model = Model(inputs = categorical_inputs + [numerical_inputs] + [lstm_inputs], outputs=out)
model.compile(optimizer='adam',loss=binary_crossentropy)
model.summary()
print('Done')

mt = market_train.iloc[train_indices]
mv = market_train.iloc[val_indices]
print('Ordering by tickers and dates')
mt = mt.sort_values(['assetCode', 'time'], ascending=[True, True])
mv = mv.sort_values(['assetCode', 'time'], ascending=[True, True])
m = appendNewDays(market_train)
del market_train
gc.collect()

mytrain = SequenceGenerator(mt, num_cols, cat_cols, lstm_cols, window = windowsize, train=True)
myvalid = SequenceGenerator(mv, num_cols, cat_cols, lstm_cols, window = windowsize, train=True)
del mt
del mv
gc.collect()
train_steps = mytrain.steps()
test_steps = myvalid.steps()
print('Done')

from keras.callbacks import EarlyStopping, ModelCheckpoint
print('Train the Neural Network Model...')
check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5,verbose=True)
mytime = time.time()
model.fit_generator(
          generator = mytrain,
          validation_data = myvalid,
          max_q_size = 2,
          #pickle_safe = False,
          callbacks=[early_stop,check_point], workers=2, use_multiprocessing=True)
fitting_time = time.time() - mytime
print('Fitting time took a total time of:', fitting_time)
print('Done')
          #epochs=1,
          #steps_per_epoch=train_steps, 
          #validation_steps=test_steps,
          #mytrain.generate(),
          #validation_data=myvalid.generate(),

if False:
    def get_input(market_train):
        y = (market_train['returnsOpenNextMktres10'] >= 0).values
        r = market_train['returnsOpenNextMktres10'].values
        u = market_train['universe']
        d = market_train['time'].dt.date
        return y, r,u,d
    
    # r, u and d are used to calculate the scoring metric
    y_valid, r_valid,u_valid,d_valid = get_input(myvalid.data)
    gc.collect()
    
    print('Evaluating confidence that will be used as submission...')
    model.load_weights('model.hdf5')
    confidence_valid = np.zeros(myvalid.data.shape[0] + myvalid.batch_size)
    flag = 0
    for myX in myvalid.generate():
        preds = model.predict(myX[0])[:,0] * 2 -1
        confidence_valid[flag:flag+len(preds)] = preds
        flag += len(preds)
    confidence_valid = confidence_valid[0:len(y_valid)]
    print(accuracy_score(confidence_valid>0,y_valid))
    plt.hist(confidence_valid, bins='auto')
    plt.title("predicted confidence")
    plt.show()
    print('Done.')
    
    # calculation of actual metric that is used to calculate final score
    print('Calculating actual metric that is used to calculate the final score...')
    r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
    x_t_i = confidence_valid * r_valid * u_valid
    data = {'day' : d_valid, 'x_t_i' : x_t_i}
    df = pd.DataFrame(data)
    x_t = df.groupby('day').sum().values.flatten()
    mean = np.mean(x_t)
    std = np.std(x_t)
    score_valid = mean / std
    print('Done')
    print(score_valid)

days = env.get_prediction_days()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
remember_days = []

print('starting the real predictions')
# market_obs_df, news_obs_df, predictions_template_df = next(days)
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print('working on day:', n_days)
    t = time.time()
    print('Calling the merging function')
    market_obs_df = merge_market_news(market_obs_df, news_obs_df)
    ###########################
    for i, cat in enumerate(cat_cols):
        market_obs_df[cat] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    #market_obs_df[num_cols] = market_obs_df[num_cols].fillna(market_obs_df[num_cols].mean())
    market_obs_df[all_cols] = market_obs_df[all_cols].fillna(0)
    market_obs_df[num_cols] = scaler.fit_transform(market_obs_df[num_cols])
    market_obs_df[lstm_cols] = scaler.fit_transform(market_obs_df[lstm_cols])
    print('done cleaning the data')
    
    X_num_test = market_obs_df[num_cols].values
    X_test = {'num':X_num_test}
    
    m = appendNewDays(m, market_obs_df)
    
    lstmlist = []
    for company in market_obs_df['assetCode']:
        lstm = m[m['assetCode'] == company]
        for _ in range(windowsize - lstm.shape[0]): # Repeat values if there's missing history
            lstm = lstm.iloc[0:1].append(lstm)
        lstmlist.append(lstm[lstm_cols].iloc[0:windowsize].values)
    X_test['lstm'] = np.array(lstmlist)
    print('categorical variables next')
    for cat in cat_cols:
        X_test[cat] = market_obs_df[cat].values
    print('data ready with day:', n_days)
    prep_time += time.time() - t
    
    t = time.time()
    
    market_prediction = model.predict(X_test)[:,0] * 2 -1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() -t
    print('predictions done')
    
    t = time.time()
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction})
    # insert predictions to template
    predictions_template_df['blah'] = preds['confidence']
    predictions_template_df = predictions_template_df.drop('confidenceValue', axis=1).fillna(0).rename(columns={'blah':'confidenceValue'})
    #predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
total = prep_time + prediction_time + packaging_time + fitting_time
print(f'Fitting Time: {fitting_time:.2f}s')
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')

# distribution of confidence as a sanity check: they should be distributed as above
plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()


confidence_valid = [model.predict(myX[0])[:,0]*2 -1 for myX in myvalid.generate()]
print('done')