# <center>US January Flight Delay Prediction Project</center>

## <center>(Source: Kaggle Datasets)</center>
## <center>Notebook: [ML-Jan Flight Prediction](https://github.com/haorzeng1997/Jan-Flight-Delay-Prediction/blob/master/Flight-Delay-Prediction-Notebook.ipynb)</center>

- Created a classification model (f1 score: **0.75** & accuracy score: **0.94**) to predict whether a flight would delay or not based on features including distance, departure airports, arrival airports and etc.

- Utilized 1.19 million pieces of flight information to build the model and conduct analysis.

- Engineered features from text and categorical data to boost efficiency.

- Optimized Decision Tree and Random Forest models using RandomizedSearchCV.

- Evaluated models using f1 score when dealing with imbalanced classes.

## Resources Used

Python Version: 3.8

Packages: pandas, numpy, sklearn, matplotlib, seaborn

## Dataset Description
Gathered US Flight Data for the month of Jan 2019 and Jan 2020 from Kaggle dataset. 
According to the author of the dataset, This data is collected from the Bureau of Transportation Statistics, Govt. of the USA. This data is open-sourced under U.S. Govt. Works. This dataset contains all the flights in the month of January 2019 and January 2020. There are more than 400,000 flights in the month of January itself throughout the United States. The features were manually chosen to do a primary time series analysis. There are several other features available on their website. This data could well be used to predict the flight delay at the destination airport specifically for the month of January in upcoming years as the data is for January only.

- DAY_OF_MONTH: Day of the month
- DAY_OF_WEEK: Day of week starting from Monday
- OP_UNIQUE_CARRIER: Unique Carrier Code
- OP_CARRIER_AIRLINE_ID: An identification number assigned by US DOT to identify a unique airline (carrier)
- OP_CARRIER: Code assigned by IATA and commonly used to identify a carrier
- TAIL_NUM: Tail Number
- OP_CARRIER_FL_NUM: Flight Number
- ORIGIN_AIRPORT_ID: Origin Airport, Airport ID
- ORIGIN_AIRPORT_SEQ_ID: Origin Airport, Airport Sequence ID
- ORIGIN: Origin Airport
- DEST_AIRPORT_ID: Destination Airport, Airport ID
- DEST_AIRPORT_SEQ_ID: Destination Airport, Airport Sequence ID
- DEST: Destination Airport
- DEP_TIME: Actual Departure Time (local time: hhmm)
- DEP_DEL15: Departure Delay Indicator, 15 Minutes or More (1=Yes, 0=No)
- DEP_TIME_BLK: Departure Time Block, Hourly Intervals
- ARR_TIME: Actual Arrival Time (local time: hhmm)
- ARR_DEL15: Arrival Delay Indicator, 15 Minutes or More (1=Yes, 0=No)
- CANCELLED: Cancelled Flight Indicator (1=Yes, 0=No)
- DIVERTED: Diverted Flight Indicator (1=Yes, 0=No)
- DISTANCE: Distance between airports (miles)

The dataset contains 1191331 pieces of flight information and 23 features.

## Data Cleaning
To prepare for EDA and model building, I cleaned the dataset:

- NULL values

	"Unnamed: 21" column is an empty column, so I dropped the column.
	
	For flights that were labelled as "cancelled" and "diverted", their ARR_TIME and ARR_DEL15 are empty. This makes sense since once a flight was cancelled or diverted, we won't be able to collect information about arriving time. 26100 flights contain missing values. Since the dataset contains more than 1 million flights information, I decided to drop flights with missing values.
	
- Drop highly correlated columns

	Flight #: In the dataset, TAIL_NUM and OP_CARRIER_FL_NUM represent a same thing-the flight number. So I dropped TAIL_NUM.
	
	Carrier: OP_UNIQUE_CARRIER, OP_CARRIER_AIRLINE_ID and OP_CARRIER represent the carrier of flight. I kept OP_CARRIER and dropped the other two. Since OP_CARRIER is more readable (for example, AA, UA etc.) 

	Origin and Destination: ORIGIN_AIRPORT_ID, ORIGIN_AIRPORT_SEQ_ID and ORIGIN indicate origin airport. DEST_AIRPORT_ID, DEST_AIRPORT_SEQ_ID, DEST indicate the destination airport. Among them, DEST and ORIGIN are the most common used ones (for example, ORD, ATL etc.) Thus I kept DEST and ORGIN and dropped others.

## Exploratory Data Analysis

- Flight delay is related to many factors. Generally, the most frequent reason is the delay of departure. Based on historical data (Jan 2019 and Jan 2020 data), if a flight was departure delay, there was 77.74% chance that the flight delayed upon arrival. 

- On what days of January travelers can expect arrival delay? (Total number of delays by Day of month)

	Surprisingly, the pattern is not obvious. I assume that weather condition plays an important role here. But we can see that for 2019, the delay chance was higher in the second half of January and for 2020, the delay chance was higher in the first half of January. Due to the COVID-19, people can reasonable assume that flight numbers decreased during the second half of January 2020.
![alt text][logo1]

[logo1]: https://github.com/haorzeng1997/Jan-Flight-Delay-Prediction/blob/master/graph/dom.png "dom"

- On what days of week travelers can expect arrival delay? (Total number of delays by Day of week)

	In Jan 2019, most of arrival delays happened on Wednesdays and Thursdays. In Jan 2020, most of arrival delays happened on Thursdays, Fridays, and Saturdays.
![alt text][logo2]

[logo2]: https://github.com/haorzeng1997/Jan-Flight-Delay-Prediction/blob/master/graph/dow.png "dow"

- Flights departed from/arrived at what airports are more likely to delay?
	
	Flights departed from ORD, DFW and ATL are more likely to experience departure delay.
	Flights arrived at ORD, DFW, ATL and LGA are more likely to experience arrival delay.
	In conclusion, ORD, DFW and ATL are busiest airports in the United States, if a passenger's destination or origin airport is one of them, the passenger are very likely to experience arrival or departure delay.
![alt text][logo3]

[logo3]: https://github.com/haorzeng1997/Jan-Flight-Delay-Prediction/blob/master/graph/top10airport.png "top10airport"

- Flights of what carriers are more likely to delay?

	WN, AA and OO (Southwest Airlines, American Airlines and SkyWest Airlines) are the top 3 carriers have the highest total number of delayed flights both in 2019 and 2020.
![alt text][logo4]

[logo4]: https://github.com/haorzeng1997/Jan-Flight-Delay-Prediction/blob/master/graph/top10carrier.png "top10carrier"

- Total number of flights in Jan 2019 and Jan 2020

	I mentioned that COVID-19 may cause total number of flights decreased. However, my assumption may be  wrong. There were more flights in Jan 2020 than Jan 2019.
![alt text][logo5]

[logo5]: https://github.com/haorzeng1997/Jan-Flight-Delay-Prediction/blob/master/graph/totalnumberflight.png "flightversus"

## Feature Engineering

- ARR_TIME_BLK: I created this column using the same logic of DEP_TIME_BLK. 

- Dummy variables: I changed the dtype of "ORIGIN", "DEST", "DEP_TIME_BLK", "ARR_TIME_BLK", "YEAR", "DEP_DEL15", "ARR_DEL15" to categorical. Then I used pandas.get_dummies to create dummy variables for these categorical data.

## Model Building
I planned to build a decent classification model based on Jan 2019 and Jan 2020 US flight data to predict whether a flight would delay or not. Future travelers and agencies could use this model to predict the chance of delay of a particular flight. 

I performed:

-   Decision Tree
-   Random Forest
  
I would use  **"f1 score"**  to score each model. Because the classes are imbalanced, f1 score is a better benchmark than accuracy score.

## Model Performances

The Random Forest outperformed the Decision Tree on the test and validation sets.

- Random Forest: ~0.75 (f1 score)
- Decision Tree: ~0.7 (f1 score)

## Notes

Due to the computing power limitation, the SVC and AdaBoost algorithms failed to generate any result. The Random Forest and Decision Tree models already used half of the data (more than 500,000 pieces of flight info). Thus, the potential of increasing models' accuracy is high.
