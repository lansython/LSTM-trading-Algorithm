# This is a sample script implementing a simple momentum trading algorithm
# We compute 2-week (10-day) return. If it is positive, we go long on the stock
# If it is negative, we go short on the stock. We rebalance positions so that 
# total position is at most the current wealth each day

library(putils) 
num_symbols <- 100
lookback <- 10 # Compute momentum using past 10 day return

# Function to initialise the trading state object
initialise_state <- function(data){
  dates <- sort(as.Date(unique(data$date)))
  symbols <- sort(unique(data$symbol))
  
  # Store past 10-day prices (rows = time lags, cols = symbols)
  lagged_price <- matrix(NA, lookback, num_symbols)
  colnames(lagged_price) <- symbols
  rownames(lagged_price) <- paste0('lag_', 1:lookback)
  
  for (i in 1:lookback){
    date <- dates[length(dates) - i + 1] #Select lasgt 10 dates of training data 
    sub_df <- data[data$date == date, ] # FIlter data
    lagged_price[i, sub_df$symbol] <- sub_df$close 
  }
  
  # Positions (current dollar allocation per symbol)
  positions <- setNames(rep(0, num_symbols), symbols) # named vector
  
  state <- list(lagged_price=lagged_price, wealth=1, positions=positions)
  return(state)
}

# Function to execute the trading algorithm
trading_algorithm <- function(new_data, state){
  # Extract state variables
  bunch(lagged_price, wealth, positions) %=% state
  
  # Update lagged price matrix
  lagged_price[2:lookback, ] <- lagged_price[1:(lookback - 1), ]
  lagged_price[1, new_data$symbol] <- new_data$close
  
  # Update wealth and positions to reflect 1-day PnL
  r1d <- lagged_price[1, ] / lagged_price[2, ] - 1 # 1-day return VECTOR FOR EACH STOCK
  wealth <- wealth + sum(positions * r1d) 
  positions <- positions * (1 + r1d)
  
  # Compute momentum (sign of return in 10 days) to construct new positions
  momentum <- sign(lagged_price[1, ] - lagged_price[10, ])
  new_positions <- wealth / num_symbols * momentum
  
  # Compute trade adjustments and new state
  trades <- new_positions - positions
  new_state <- list(lagged_price=lagged_price, wealth=wealth, positions=new_positions)
  
  return(list(trades=trades, new_state=new_state))
}


