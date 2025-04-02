### This is a script used to evaluate the performance of a trading algorithm
library(putils) 
num_symbols <- 100

#### Main walk-forward backtesting function ####
walk_forward <- function(strategy, initialiser, df_train, df_test){
  # Initialise state
  state <- initialiser(df_train)
  unique_test_dates <- sort(unique(df_test$date))
  n_test_dates <- length(unique_test_dates)
  positions <- rep(0, num_symbols)
  daily_pnl <- rep(0, n_test_dates)
  
  for (i in 1:n_test_dates){
    # Extract new data for the current test date
    new_data <- df_test[df_test$date == unique_test_dates[i], ]
    
    # Run the trading strategy
    bunch(trades, state) %=% strategy(new_data, state)
    
    # Update position and daily PnL
    if (i == 1){
      price <- new_data[, 'close']
      positions <- trades
      daily_pnl[i] <- 0 # No PnL on the first day since initial position = 0
    } else {
      price_lag1 <- price
      price <- new_data[, 'close']
      r1d <- price / price_lag1 - 1 # compute simple return
      daily_pnl[i] <- sum(positions * r1d) # compute daily PnL
      positions <- positions * (1 + r1d) + trades # update position
    }
    printPercentage(i, n_test_dates)
  }
  
  # Compute cumulative wealth
  wealth_seq <- 1 + cumsum(daily_pnl)
  
  # Ensure wealth does not go negative
  if (any(wealth_seq <= 0)) {
    first_idx <- which(wealth_seq <= 0)[1]
    wealth_seq[first_idx:n_test_dates] <- 0
  }
  
  # Plot wealth evolution
  plot(wealth_seq, type='l')
  
  return(wealth_seq)
}

#### Use this part to test that your script is working properly ####

# load data
df<- read.csv('df_train.csv')

# here we use part of the df_train.csv for testing, 
# an additional test data set will be supplied in actual running
train_idx <- df$date < as.Date('2012-01-01')
df_train <- df[train_idx, ]
df_test <- df[!train_idx, ]

# Load strategy functions
script_name <- 'Project.R' # change to your own script name
source(script_name)

# Run backtest
wealth_seq <- walk_forward(trading_algorithm, initialise_state, df_train, df_test)

# Print log final wealth
println('log wealth = ', log(tail(wealth_seq, 1)))

