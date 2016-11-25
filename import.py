import csv, json, sys
from dateutil.parser import parse
from datetime import datetime
import pytz
from sklearn import preprocessing

# Used to stop us getting loans that aren't expired yet.
DATE_OF_DUMP = '2016-11-24T17:51:04Z'

# Number of first file to look at. Currently set to begining of 2016
FIRST_FILE = 2037
LAST_FILE = 2271

# Gets every xTh file, use 1 to use every file.
FILE_INCREMENT = 50

OUTPUT_TRAIN = 'output/train_rough_k.csv'

# Feature ideas: 
#  is photo person smiling
#  is country on terror watch list
#  did the country have an international disaster in the last 2 months
#location.country
MANUAL_KEYS = ['expired', 'description_length', 'description_sentiment', 'country', 'gender', 'pictured']
# 'id','status', 'funded_amount',  'planned_expiration_date'
KEYS = ['sector', 'loan_amount','partner_id']

INTERESTING_COUNTRIES =['Tajikistan', 'Albania', 'Armenia', 'Azerbaijan',
					'Kyrgyzstan', 'Kosovo', 'Mongolia', 'Turkey','Ukraine']


def readFile(fileName):
	input = open(fileName)
	data = json.load(input)
	input.close()

	# data rows  
	for obj in data['loans']:
		row = []

		if 'planned_expiration_date' in obj and not obj['planned_expiration_date'] == None:
			dateParsed = parse(obj['planned_expiration_date'])
			now = parse(DATE_OF_DUMP)
			# if expiry date hasn't been reached skip it since it's outcome is unkown.
			if now < dateParsed:
				print dateParsed
				break #did continue, but this is faster and we don't lost much data.

		# Filter to interesting countries
		if INTERESTING_COUNTRIES:
			if not obj['location']['country'] in INTERESTING_COUNTRIES:
				continue

		# are expired ones are still in fundraising state.
		# TODO: find API docs on statuses and figure out diff between expired, refunded, etc.
		expired = obj['status'] == 'expired'
			#(obj['status'] == 'fundraising' # Don't think should have any in fundraising state...
			#or obj['status'] == 'inactive_expired' # These ones might have never been posted.
			#or obj['status'] == 'expired')
		row.append(expired)		

		# Description length
		if 'en' in obj['description']['texts']:			
			row.append(len(obj['description']['texts']['en'].encode('utf8', 'replace')))
			# some dumps have no description even though the API returns a description, particularly for delinquent or defaulted loans:
			if len(obj['description']['texts']['en'].encode('utf8', 'replace')) == 0:
				print "Missing desc in dump: http://api.kivaws.org/v1/loans/" + str(obj['id']) + ".json"
				print str(obj['description']['texts'])
		else:
			row.append(0)

		# Country
		row.append(obj['location']['country'])

		# Gender
		row.append(obj['borrowers'][0]['gender']) #"pictured":true

		# Pictured
		row.append(obj['borrowers'][0]['pictured'])

		# Add all automatic keys.
		for key in KEYS:
			if isinstance(obj[key], basestring):
				obj[key] = obj[key].encode('utf8', 'replace')
			# coerse partner_id into string so we don't accidentally do math on it.
			if key == 'partner_id':
				obj[key] = 'p_' + str(obj[key])
			row.append(obj[key])
		csv_file.writerow(row)


# Main code:
output = open(OUTPUT_TRAIN, "w")
csv_file = csv.writer(output)

#Header
row = MANUAL_KEYS
for key in KEYS:
	row.append(key)

csv_file.writerow(row)

# Iterate over files
for i in range(FIRST_FILE, LAST_FILE, 1):
	print "Read " +  str(i) 
	readFile("kiva_ds_json/loans/" + str(i) + ".json")





'''
with open("data.json") as file:
    data = json.load(file)

with open("data.csv", "w") as file:
    csv_file = csv.writer(file)
    for item in data:
        csv_file.writerow([item['pk'], item['model']] + item['fields'].values())
        '''