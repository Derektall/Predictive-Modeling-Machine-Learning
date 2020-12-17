library(data.table)
library(tidyverse)
library(ggplot2)
library(prophet)
library(readxl)


## I used hourly weather data from kaggle here: https://www.kaggle.com/selfishgene/historical-hourly-weather-data?select=pressure.csv 
## The goal is to use Facebook Prophet to predict temperature in one city (Vancouver).
## The great thing about Prophet is that it is robust to missing values,
## multiple seasonality (which this data has as i'll show in a second), and outliers.

##First i'll use a basic Prophet forecast with no tuning. Then I'll start 
## adding regressors and tuning parameters in order to try and find the best 
## cross-validation score

WeatherData <- read_excel("C://Users//Zoidb//Downloads//Portland_Weather_Forecasts.xlsx", col_names = TRUE)
##I need to create separate datasets for the
WeatherData$DateTime <-  as.Date(as.POSIXct(WeatherData$DateTime, format = '%Y-%m-%d %H:%M:%S'))

##First i'll make a quick plot to show how the data is behaving

ggplot(data=WeatherData, aes(x=DateTime, y=Temperature))+
  geom_line()

##We see that there are multiple seasonalities involved here.

##Next i'll need to change DateTime to "ds" and Temperature to "y" as required by Prophet

WeatherData <- mutate (
  WeatherData,
  ds = DateTime,  
  y = Temperature  
)

##Loading the data into prophet
Model <- prophet(WeatherData)

##Creating future dataframe to make predictions
future <- make_future_dataframe(Model, periods = 240, freq="h") ##10 days * 24 hours = 240 periods
##Creating the forecast
forecast <- predict(Model, future)

plot(Model, forecast)
##We can see that in the next 3 months the temperatures will
## increase overall

prophet_plot_components(Model, forecast)
##We can look at the trend and seasonality and see some patterns
## like how the temperature tends to increase in summer months (not surprising)
## and also tends to dip in the morning and afternoon.

##Now we need to cross-validate (take different cutoff points and test the
## model against the actual values during the horizon)

prophet_plot_components(Model, forecast)

##I'll start with 9000 observations (a little over 2 years) as training data and train the model every 5000 hours for a horizon similar to the forecast
Model.cv <- cross_validation(Model, initial = 20000, period = 10000, horizon = 2160, units = 'hours') 
Model_stats <- performance_metrics(Model.cv)
##I'll take the mean MAE as my benchmark for this model
mean(Model_stats$mae) #4.8018, which means that the forecast is off on average
## around 4.8 degrees. 

