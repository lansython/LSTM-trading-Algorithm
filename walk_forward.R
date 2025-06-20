### This is a script used to evaluate the performance of a trading algorithm
library(putils) 
num_symbols <- 100

#### Main walk-forward backtesting function ####
walk_forward <- function(strategy, initialiser, df_train, df_test){
  # Initialise state
  state <- initialiser(df_train)
  unique_test_dates <- sort(unique(df_test$date))
  n_test_dates <- length(unique_test_dates)
  positions <- matrix(0, nrow = n_test_dates, ncol = num_symbols)
  daily_pnl <- rep(0, n_test_dates)
  
  for (i in 1:n_test_dates){
    # Extract new data for the current test date
    new_data <- df_test[df_test$date == unique_test_dates[i], ]
    
    # Run the trading strategy
    bunch(trades, state) %=% strategy(new_data, state)
    
    # Update position and daily PnL
    if (i == 1){
      price <- new_data[, 'close']
      positions[i, ] <- trades
      daily_pnl[i] <- 0 # No PnL on the first day since initial position = 0
    } else {
      price_lag1 <- price
      price <- new_data[, 'close']
      r1d <- price / price_lag1 - 1 # compute simple return
      daily_pnl[i] <- sum(positions * r1d) # compute daily PnL
      positions[i, ] <- positions[i - 1, ] * (1 + r1d) + trades# update position
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
  # compute portfolio Sharp ratio, alpha and beta
  rf <- 0.03 # risk free rate
  # Annualised Sharpe ratio
  SR <- mean(daily_pnl - rf/252, na.rm=TRUE) / sd(daily_pnl, na.rm=TRUE) * sqrt(252)
  cat('Sharpe ratio =', SR)
  return(wealth_seq)
}

#### Use this part to test that your script is working properly ####

# load data
df<- read.csv('df_train.csv')

# here we use part of the df_train.csv for testing, 
# an additional test data set will be supplied in actual running
train_idx <- df$date < as.Date('2013-01-01')
df_train <- df[train_idx, ]
df_test <- df[!train_idx, ]

# Load strategy functions
script_name <- 'lightgbm_trading.R' # change to your own script name
source(script_name)

# Run backtest
wealth_seq= walk_forward(trading_algorithm, initialise_state, df_train, df_test)

# Print log final wealth
println('log wealth = ', log(tail(wealth_seq, 1)))

# Compute daily returns
daily_returns <- diff(wealth_seq) / head(wealth_seq, -1)

# Drop any NA (e.g., first day)
daily_returns <- daily_returns[!is.na(daily_returns)]




# Annualized return
total_days <- length(daily_returns)
annualized_return <- (tail(wealth_seq, 1))^(252 / total_days) - 1

# Annualized volatility
annualized_volatility <- volatility * sqrt(252)

# Max drawdown
running_max <- cummax(wealth_seq)
drawdowns <- 1 - wealth_seq / running_max
max_drawdown <- max(drawdowns, na.rm = TRUE)

# Cumulative return
cumulative_return <- tail(wealth_seq, 1) - 1

# Print all metrics
cat("\n===== Performance Metrics =====\n")
cat(sprintf("Final Wealth: %.4f\n", tail(wealth_seq, 1)))
cat(sprintf("Cumulative Return: %.2f%%\n", cumulative_return * 100))
cat(sprintf("Annualized Return: %.2f%%\n", annualized_return * 100))
cat(sprintf("Annualized Volatility: %.2f%%\n", annualized_volatility * 100))
cat(sprintf("Max Drawdown: %.2f%%\n", max_drawdown * 100))
cat("================================\n")

  
library(quantmod)
getSymbols('SPY', from='2013-01-01', to='2014-01-01')
SPY_price <- SPY$SPY.Adjusted
library(xts)

# Step 1: Scale strategy wealth
wealth_scaled <- wealth_seq / as.numeric(wealth_seq[1]) * 100

# Step 2: Create xts version of wealth, aligned with df_test dates
wealth_dates <- sort(unique(df_test$date))
wealth_xts <- xts(wealth_scaled[1:length(wealth_dates)], order.by = as.Date(wealth_dates))

# Step 3: Scale SPY prices
SPY_scaled <- SPY$SPY.Adjusted / as.numeric(SPY$SPY.Adjusted[1]) * 100

# Step 4: Align both time series to common index
combined <- merge(wealth_xts, SPY_scaled, join = "inner")
colnames(combined) <- c("Strategy", "SPY")

# Step 5: Plot
plot(combined, legend.loc = 'topleft', main = 'Wealth vs SPY', col = c("blue", "red"))


#detach("package:dplyr", unload = TRUE)
# Compute strategy daily returns from wealth
# Compute daily returns (already xts)
daily_returns <- diff(wealth_xts) / stats::lag(wealth_xts, k = 1)
strategy_returns_xts <- daily_returns  # it's already xts!
# Get SPY adjusted prices and compute log returns
SPY_ret_xts <- diff(log(SPY_price))

# Align dates and compute correlation
aligned_returns <- merge(strategy_returns_xts, SPY_ret_xts, join = "inner")
colnames(aligned_returns) <- c("Strategy", "SPY")

# Correlation calculation
correlation <- cor(aligned_returns$Strategy, aligned_returns$SPY, use = "complete.obs")
cat(sprintf("Correlation with SPY: %.4f\n", correlation))

# Print log final wealth
min(wealth_seq)

