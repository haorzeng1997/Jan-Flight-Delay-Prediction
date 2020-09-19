# <center>US January Flight Delay Prediction Project</center>

## <center>(Source: Kaggle Datasets)</center>
## <center>Notebook: [ML-DS Salary Estimate](https://github.com/haorzeng1997/Data-Scientist-Salary-Project/blob/master/data%20science%20salary%20estimate%20project%20report.pdf)</center>

- Created a classification model (f1 score: 0.75 & accuracy score: 0.94) to predict whether a flight would delay or not based on features including distance, departure airports, arrival airports and etc.

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

 
