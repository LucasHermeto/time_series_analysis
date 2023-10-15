import os
import pandas as pd
import pickle

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesModel():
    def __init__(self, version_name, df, label_str,
                 all_models_params, index_cols, all_features=None):
        self.version_name = version_name
        self.df = df
        self.label_str = label_str
        self.all_models_params = all_models_params
        self.index_cols = index_cols
        self.features = [
            feature for feature in all_features
            if feature not in index_cols + [label_str]
        ]
        self.df_all_predictions = None
        self.date_str = datetime.now().strftime('%Y_%m_%d')

    
    def execute(self, min, max):
        print('Running Models')
        self.run_all_models(min, max)

        print('Saving all models')
        self.save_all_models()

        print('Saving features')
        self.save_features()

        print('Saving all models prediction')
        self.save_all_predictions()

        print('Success!')


    def run_all_models(self, min, max):
        X_train, X_test, y_train, y_test = self.split_train_test(min,
            max)
        self.df_all_predictions = pd.DataFrame(columns=self.index_cols +
            ['cd_model', 'cd_sample', 'nu_predict', 'nu_label'])
        
        for model_name in self.all_models_params:
            print('Training model:', model_name)

            model_prediction = self.run_model(model_name, X_train, X_test,
                                              y_train, y_test)
            self.df_all_predictions = pd.concat([
                model_prediction,self.df_all_predictions
            ])
            print('-----------------------------------------------------------')
        
        return self.df_all_predictions
    
    def split_train_test(self, min, max):
        df_train, df_test = self._split_train_test_model(min, max)
        
        df_train.set_index(self.index_cols, inplace=True)
        df_test.set_index(self.index_cols, inplace=True)

        X_train = df_train[self.features]
        y_train = df_train[self.label_str]
        X_test = df_test[self.features]
        y_test = df_test[self.label_str]

        return X_train, X_test, y_train, y_test

    
    def _split_train_test_model(self, min, max):
        df_model = self.df[self.df['YEAR_MONTH'] >= min].copy()
        df_model = df_model[df_model['YEAR_MONTH'] < max]

        test_year_month = df_model['YEAR_MONTH'].max()

        df_model_train = df_model[df_model['YEAR_MONTH'] < test_year_month]
        df_model_test = df_model[df_model['YEAR_MONTH'] == test_year_month]

        return df_model_train, df_model_test

    def run_model(self, model_name, X_train, X_test, y_train, y_test):
        model = self.all_models_params[model_name]['model']

        X_train_scaled, X_test_scaled = self._get_X_transformed(model_name, 
            X_train, X_test)
        model_predictions = self._fit_predict(model, X_train_scaled,
            X_test_scaled, y_train, y_test)
        model_predictions.loc[:, 'cd_model'] = model_name

        return model_predictions
    
    def _get_X_transformed(self, model_name, X_train, X_test):
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        if self.all_models_params[model_name]['scaler']:
            if 'scaler_function' in self.all_models_params[model_name]:
                scaler = self.all_models_params[model_name]['scaler_function']
            else:
                scaler = MinMaxScaler()
            
            X_train_scaled[:] = scaler.fit_transform(X_train)
            X_test_scaled[:] = scaler.fit_transform(X_test)
        
        if self.all_models_params[model_name]['encoded']:
            all_object_columns = (X_train_scaled
                .select_dtypes(include='object').columns
            )

            for column in all_object_columns:
                X_train_scaled[column] = X_train_scaled[column].astype(
                    'category')
                X_train_scaled[column] = X_train_scaled[column].cat.codes

                X_test_scaled[column] = X_test_scaled[column].astype(
                    'category')
                X_test_scaled[column] = X_test_scaled[column].cat.codes
            
        return X_train_scaled, X_test_scaled

    
    def _fit_predict(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        X_train_temp = self._predict_model(model, X_train, y_train)
        X_train_temp.loc[:, 'cd_sample'] = 'train'
        X_train_temp = X_train_temp[[
            'nu_predict', 'nu_label', 'cd_sample'
        ]].reset_index()

        X_test_temp = self._predict_model(model, X_test, y_test)
        X_test_temp.loc[:, 'cd_sample'] = 'test'
        X_test_temp = X_test_temp[[
            'nu_predict', 'nu_label', 'cd_sample'
        ]].reset_index()

        model_predictions = pd.concat([X_test_temp, X_train_temp])
        return model_predictions
    

    def _predict_model(self, model, X, y):
        X_model = X.copy()
        X_model.loc[:, 'nu_predict'] = model.predict(X_model)
        X_model.loc[:, 'nu_label'] = y

        return X_model


    def save_all_models(self):
        for model_name in self.all_models_params:
            model = self.all_models_params[model_name]['model']

            model_path = f'../models/{self.date_str}/{self.version_name}/trained/'

            path_exists = os.path.exists(model_path)
            if path_exists == False:
                os.makedirs(model_path)
            
            if ('save_catboost' in self.all_models_params[model_name] and
                self.all_models_params[model_name]['save_catboost']):
                model_path += f'{model_name}.cbm'
                self._save_catboost(model, model_path)
            else:
                model_path += f'{model_name}.pickle'
                self._save_pickle(model, model_path)
            
            print(f'Model {model_path} saved.')
    

    @staticmethod
    def _save_catboost(model, path):
        model.save_model(path)


    @staticmethod
    def _save_pickle(object, path):
        pickle.dump(object, open(path, 'wb'))


    def save_features(self):
        all_features = {
            'features': self.features,
            'index_cols': self.index_cols,
            'feature_importance': self._get_feature_importance()
        }
        feature_path = f'../models/{self.date_str}/{self.version_name}/features.pickle'

        self._save_pickle(all_features, feature_path)
    

    def _get_feature_importance(self):
        sort_dict = {}
        for model_name in self.all_models_params:
            model = self.all_models_params[model_name]['model']
            try:
                marklist = dict(sorted(
                    dict(zip(self.features, model.feature_importances_)).items(),
                    key=lambda x:x[1], reverse=True
                ))
            except:
                marklist = None
            
            model_marklist = {model_name: marklist}
            sort_dict.update(model_marklist)
        
        return sort_dict

    
    def save_all_predictions(self):
        predictions_path = f'../models/{self.date_str}/{self.version_name}/predictions/'

        path_exists = os.path.exists(predictions_path)
        if path_exists == False:
            os.makedirs(predictions_path)
        
        try:
            print('Trying to save as parquet')
            predictions_path += 'df_all_predictions.parquet'
            self.save_file(self.df_all_predictions, predictions_path)
        except:
            print('Error. Saving as csv')
            predictions_path = predictions_path.replace('parquet', 'csv')
            self.save_file(self.df_all_predictions, predictions_path)
        
    
    def save_file(self, df, path, mode='w'):
        file_type = path.split('.')[-1]

        if file_type == 'csv':
            df.to_csv(path, mode=mode)
            print(f'File {path} saved.')
        elif file_type == 'parquet':
            if mode == 'a':
                df_temp = pd.read_parquet(path)
                df = pd.concat([df, df_temp])
            else:
                if mode != 'w':
                    print('Not valid mode, please use \'a\' or \'w\'')
                    return
            
            df.to_parquet(path)
            print(f'File {path} saved.')
        else:
            print('Not valid file type, please use \'csv\' or \'parquet\'')
