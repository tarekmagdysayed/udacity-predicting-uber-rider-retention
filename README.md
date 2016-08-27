# Predict Uber Rider Retention
## Background:
Uber is interested in predicting rider retention. To help explore this question, Uber provided a sample dataset of a cohort of users who signed up for an Uber account in January 2014. The data was pulled several months later; we consider a user retained if they were “active” (i.e. took a trip) in the preceding 30 days.

I would use the data set to help understand what factors are the best predictors for retention, and offer suggestions to operationalize those insights to help Uber.

## Dataset description:
* city: city this user signed up in
* phone: primary device for this user
* signup_date: date of account registration; in the form ‘YYYY­MM­DD’
* last_trip_date: the last time this user completed a trip; in the form ‘YYYY­MM­DD’ 
* avg_dist: the average distance *(in miles) per trip taken in the first 30 days after signup 
* avg_rating_by_driver: the rider’s average rating over all of their trips 
* avg_rating_of_driver: the rider’s average rating of their drivers over all of their trips 
* surge_pct: the percent of trips taken with surge multiplier > 1
* avg_surge: The average surge multiplier over all of this user’s trips 
* trips_in_first_30_days: the number of trips this user took in the first 30 days after signing up
* uber_black_user: TRUE if the user took an Uber Black in their first 30 days; FALSE otherwise
* weekday_pct: the percent of the user’s trips occurring during a weekday