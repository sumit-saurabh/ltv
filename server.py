#!/usr/bin/python
from flask import Flask, render_template, request, redirect, Response, jsonify
#from google.cloud import bigquery
#import google.auth
from pandas.io import gbq
from lifetimes.datasets import load_transaction_data
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes import ParetoNBDFitter
from lifetimes.plotting import plot_frequency_recency_matrix
import pandas as pd
from pandas import DataFrame
import time
from werkzeug.utils import secure_filename
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
import os

# https://blog.apcelent.com/create-rest-api-using-flask.html
app = Flask(__name__, template_folder='./')
api = Api(app)
times_to_calculate_the_expectation_for = 15  # day
pickle_object_name = "trained_data.pkl"
segment_count = 5
count = 0
project_id = "fox-hackathon-0728-019"
key_file_path = "creds.json"
destination_table = 'foxhackathon_adobe_ltv.ltv_prediction_output_data'
chunk_size = 10000
customer_id_col = 'customer_id'
datetime_col = 'date'
monetary_value_col = 'transactional_value'
upload_dir = 'data'
input_file = ''


@app.after_request
def after_request(response):
	response.headers.add('Access-Control-Allow-Origin', '*')
	response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
	response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
	return response

@app.route('/predict', methods = ['GET', 'POST'])
def load_and_predict_model():
	global count
	global input_file
	if count < 2:
		load_ml_models()
		load_input_data(input_file)
		def generate():
			global count
			if count < 3:
				predict_expand_and_segment_data(bgf, ggf)
				yield "data:" + str("prediction_completed") + "\n\n"
				time.sleep(3)
				print ("Segmenting")
				count = count+1
				segment_data();
				yield "data:" + str("segmentation_completed") + "\n\n"
				time.sleep(3)
				count = count+1
				print ("Uploading data to BigQuery")
				store_dataframe_to_big_query()
				yield "data:" + str("data_uploaded") + "\n\n"
				time.sleep(3)
				count = count+1
		return Response(generate(), mimetype="text/event-stream")
	return "Done"

# def store_dataframe_to_big_query():
#         print ("Method init: store_dataframe_to_big_query")
#         global returning_customers_summary
#         start_time = time.time()
#         print ("Storing dataframe: start time = " + str(start_time))
#         returning_customers_summary.to_gbq(destination_table,
#                                          project_id,
#                                          chunksize=chunk_size,
#                                          if_exists='replace',
#                                          verbose=True
#                                          )
#         end_time = time.time()
#         print ("Storage time: " + str(end_time - start_time))
#         print ("Saved the output dataframe to bigquery.")
        
def load_ml_models():
	global bgf
	global ggf
	bgf_pickle_object_name = "bgf_trained_data.pkl"
	ggf_pickle_object_name = "ggf_trained_data.pkl"
	bgf = BetaGeoFitter()
	bgf.load_model(bgf_pickle_object_name)
	print (bgf)

	ggf = GammaGammaFitter()
	ggf.load_model(ggf_pickle_object_name)
	print (ggf)

def sanitize_data():
	print ("Method init: sanitize_data")
	global training_dataframe
	global returning_customers_summary
	global summary_with_money_value
	print ("Sanitizing dataset: Removing transaction with 0 monetary_value")
	summary_with_money_value = training_dataframe[training_dataframe['monetary_value']>0]
	print (summary_with_money_value.head())

	print ("Creating Returning Customer Summary and Filtering out non-returning customer.")
	returning_customers_summary = summary_with_money_value[summary_with_money_value['frequency']>0]
	print (returning_customers_summary.head())

@app.route('/train_model', methods = ['GET', 'POST'])
def load_and_train_model():
	global input_file
	req_data = request.get_json(force=True)
	input_file = secure_filename(req_data['fileName'])
	print ("Input file: " + input_file)
	load_input_data(input_file)
	train_model()
	global count
	count = 0
	return "Input data trained successfully."
	
def load_input_data(input_file):
	global training_dataframe
	global customer_id_col
	global datetime_col
	global monetary_value_col
	file_path = upload_dir + "/" + input_file
	print ("Loading Input Event Level Data...")
	input_dataframe = pd.read_csv(file_path)
	observation_period_end = max(input_dataframe[datetime_col])
	print ("observation_period_end: " + observation_period_end)
	print ("Input data loaded successfully.")
	print ("Input data:")
	print(input_dataframe.head())
	print ("Converting Event Level Data ---to--> User level data")
	training_dataframe = summary_data_from_transaction_data(
		input_dataframe, 
		customer_id_col,
		datetime_col,
		monetary_value_col,
		freq = 'D',
		observation_period_end = observation_period_end).reset_index()
	print('Output from summary_data_from_transaction_data')
	print(training_dataframe.head())
	agg_file = input_dataframe.groupby(customer_id_col).agg({datetime_col:'count',monetary_value_col:'sum'}).reset_index()
	agg_file['temp'] = agg_file[monetary_value_col]/agg_file[datetime_col]
	agg_file.drop([monetary_value_col,datetime_col],axis=1,inplace=True)
	agg_file.rename(columns = {'temp':'monetary_value'},inplace=True)
	print('agg_file - ')
	print(agg_file.head())
	training_dataframe = pd.merge(training_dataframe,agg_file,on=customer_id_col,how='left')
	training_dataframe.fillna(0,inplace=True)
	training_dataframe.set_index(customer_id_col,inplace=True)
	training_dataframe.to_csv('training_dataframe.csv')
	print ("Successfully transformed the data.")
	print ("User level data:")
	print(training_dataframe.head())

def sanitize_training_dataset():
	global training_dataframe
	global returning_customers_summary
	global summary_with_money_value
	print ("Sanitizing dataset: Removing transaction with 0 monetary_value")
	summary_with_money_value = training_dataframe[training_dataframe['monetary_value']>0]
	print(summary_with_money_value.head())

	print ("Creating Returning Customer Summary and Filtering out non-returning customer.")
	returning_customers_summary = summary_with_money_value[summary_with_money_value['frequency']>0]
	print (returning_customers_summary.head())

def fit_data_in_gamma_gamma_model():
	# Using GammaGammaFitter (ggf)
	global returning_customers_summary
	print ("Training: Fitting data in Gamma-Gamma Model")
	ggf = GammaGammaFitter(penalizer_coef = 0.1)  # 0.001 to 0.1 are effective (L2 regularization)
	ggf.fit(returning_customers_summary['frequency'],
					returning_customers_summary['monetary_value'])
	print (ggf)
	return ggf

def fit_data_in_bg_nbd_model():
	# Fit model
	# Frequency Model: Using BetaGeoFitter (bgf)
	global returning_customers_summary
	print ("Training: Fitting data in BG/NBD Model")
	bgf = BetaGeoFitter(penalizer_coef = 0.1)  # 0.001 to 0.1 are effective (L2 regularization)
	bgf.fit(returning_customers_summary['frequency'], returning_customers_summary['recency'], returning_customers_summary['T'])
	print (bgf)
	return bgf

def fit_data_in_pareto_nbd_model():
	# Using ParetoNBDFitter (pnbd)
	global returning_customers_summary
	print ("Training: Fitting data in ParetoNBDFitter Model")
	pnbd = ParetoNBDFitter(penalizer_coef=0.1) # 0.001 to 0.1 are effective (L2 regularization)
	pnbd.fit(returning_customers_summary['frequency'], returning_customers_summary['recency'], returning_customers_summary['T'])
	print (pnbd)
	return pnbd


def expand_column(bgf):
	global returning_customers_summary
	returning_customers_summary['historical_aov'] =  returning_customers_summary['monetary_value'] / returning_customers_summary['frequency']
	returning_customers_summary['future_aov'] = returning_customers_summary['predicted_purchase_value'] / returning_customers_summary['bgf_predicted_purchase_freq']
	returning_customers_summary['p_alive'] = bgf.conditional_probability_alive(returning_customers_summary['frequency'], returning_customers_summary['recency'], returning_customers_summary['T'])

def predict_expand_and_segment_data(bgf, ggf):
	global returning_customers_summary
	print ("PREDECTION")
	predict(bgf, ggf)

	print ("COLUMN DERIVATION")
	expand_column(bgf)

	print ("Predected data")
	print (returning_customers_summary.sort_values(by='bgf_predicted_purchase_freq').tail(1000))

def event_stream(event_name):
	yield "data:" + str(event_name) + "\n\n"

def segment_data():
	print ("Method init: segment_data")
	global returning_customers_summary
	prediction_dataframe =  returning_customers_summary.sort_values(['p_alive', 'predicted_purchase_value', 'bgf_predicted_purchase_freq'], ascending=[False, False, False])
	df = np.array_split(prediction_dataframe, segment_count)
	segments = []
	for i in range(segment_count):
		segment_name = i + 1
		df[i]["segment"] = segment_name
	returning_customers_summary = pd.concat(df)
	print ("End of method: segment_data")
	print (returning_customers_summary.head(1000))

def predict(bgf, ggf):
	# Predecting freq. using bgf
	print ("Predecting freq. using bgf")
	global returning_customers_summary
	global times_to_calculate_the_expectation_for
	sanitize_training_dataset()
	returning_customers_summary['bgf_predicted_purchase_freq'] = bgf.conditional_expected_number_of_purchases_up_to_time(
					times_to_calculate_the_expectation_for,
					returning_customers_summary['frequency'],
					returning_customers_summary['recency'],
					returning_customers_summary['T']
			)

	# # Predecting freq. using pnbd
	# print ("Predecting freq. using pnbd")
	# returning_customers_summary['pnbd_predicted_purchase_freq'] = pnbd.conditional_expected_number_of_purchases_up_to_time(
	#               times_to_calculate_the_expectation_for, 
	#               returning_customers_summary['frequency'], 
	#               returning_customers_summary['recency'], 
	#               returning_customers_summary['T']
	#       )
	# print returning_customers_summary.sort_values(by='pnbd_predicted_purchase_freq').tail(50)
	print ("Predecting value using ggf")
	returning_customers_summary['predicted_purchase_value'] = ggf.conditional_expected_average_profit(
					summary_with_money_value['frequency'],
					summary_with_money_value['monetary_value']
				)        

def train_model():
	print ("Method init: train_model")
	global training_dataframe
	global times_to_calculate_the_expectation_for
	global returning_customers_summary
	sanitize_training_dataset()
	print ("MODEL TRAINING")
	bgf = fit_data_in_bg_nbd_model()
	ggf = fit_data_in_gamma_gamma_model()
	#pnbd = fit_data_in_pareto_nbd_model()
	print ("VISUALIZATION")

	# Visualizing of Frequency/Recency Matrix
	#plot_frequency_recency_matrix(bgf)

	print ("SAVE MODEL")
	# Saving models
	print ("Saving BG/NBD Model")
	save_model(bgf, "bgf")

	print ("Saving Gamma-Gamma Model")
	save_model(ggf, "ggf")

	# print ("Saving ParetoNBDFitter Model")
	# save_model(pnbd, "pnbd")

	
def save_model(model, model_name):
	global pickle_object_name
	model_pickle_object_name = model_name + "_" + pickle_object_name
	model.save_model(model_pickle_object_name, save_data=True, save_generate_data_method=True)

@app.route('/')
def upload_file():
	return render_template('index.html')
	
@app.route('/upload', methods = ['GET', 'POST'])
def upload_file_handler():
	global upload_dir
	if request.method == 'POST':
		f = request.files['file']
		print ("++++++++++++++++++++++++++++ " + f.filename)
		f.save(os.path.join(upload_dir, secure_filename(f.filename)))
		#f.save(secure_filename(f.filename))
		return f.filename

def verify_project_google():
	global input_file
	load_and_predict_model()
	load_ml_models()
	load_input_data(input_file)
	global count
	predict_expand_and_segment_data(bgf, ggf)
	yield "data:" + str("prediction_completed") + "\n\n"
	print ("Segmenting")
	count = count+1
	segment_data()
	yield "data:" + str("segmentation_completed") + "\n\n"
	count = count+1
	print ("Uploading data to BigQuery")
	#store_dataframe_to_big_query()
	yield "data:" + str("data_uploaded") + "\n\n"
	count = count+1

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)




