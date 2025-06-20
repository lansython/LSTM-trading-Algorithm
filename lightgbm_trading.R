# This is a sample script implementing a simple momentum trading algorithm
# We compute 2-week (10-day) return. If it is positive, we go long on the stock
# If it is negative, we go short on the stock. We rebalance positions so that 
# total position is at most the current wealth each day

library(putils) 
library(dplyr)
library(quantmod)
library(lubridate)
library(lightgbm)

num_symbols <- 100
lookahead <- 21
train_length <- 504
valid_length <- 63

covariates <- c("open", "close", "low", "high", "volume", "dollar_vol",
          "dollar_vol_rank", "rsi", "atr", "natr", "macd", "sma", "ema", "roc",
          "r01", "r05", "r10", "r21", "r42", "r63", "r01dec", "r05dec", "r10dec", 
          "r21dec", "r42dec", "r63dec", "month", "weekday")

cat_var <- c('month', 'weekday')

params <- list(
  objective = 'regression',
  num_iterations = 30,
  num_leaves = 50,
  learning_rate = 0.1,
  feature_fraction = 0.60,
  bagging_fraction = 0.30,
  min_data_in_leaf = 250
)

get_features <- function(data){
  # Compute dollar vol
  data = data %>% mutate(dollar_vol=close*volume)
  
  # Compute dollar vol rank per day
  data = data %>% group_by(date) %>% mutate(dollar_vol_rank=rank(desc(dollar_vol))) %>% ungroup()
  
  # Compute RSI
  data = data %>% group_by(symbol) %>% mutate(rsi=RSI(close)) %>% ungroup()
  
  # Get Average True Range (ATR) and Normalized Average True Range (NATR)     
  data <- data %>% group_by(symbol)  %>% mutate(atr = ATR(cbind(high, low, close))[, 'atr']) %>% ungroup()
  data = data %>% group_by(symbol) %>% mutate(natr = atr / close) %>% ungroup()
  
  # Compute MACD
  data = data %>% group_by(symbol) %>% mutate(macd = MACD(close)[, 'macd']) %>% ungroup()
  
  # Compute SMA, EMA, ROC
  data <- data %>% group_by(symbol) %>% mutate(sma = SMA(close),
                                               ema = EMA(close),
                                               roc = ROC(close)) %>% ungroup()
  # Get month, day
  data <- data %>% mutate(month = month(date),
                          weekday = as.numeric(format(as.Date(date), "%u")))
  
  T = c(1, 5, 10, 21, 42, 63)
  # Compute Lag returns
  for (t in T) {
    data = data %>% group_by(symbol) %>% mutate(!!sprintf("r%02d", t) := (close / lag(close, t)) - 1) %>% ungroup()
  }
  # Compute Lag returns deciles
  for (t in T) {
    data = data %>% group_by(date) %>%mutate(!!sprintf("r%02ddec", t) := ntile(.data[[sprintf("r%02d", t)]], 10)) %>% ungroup()
  }
  # Compute Forward returns, not used for prediction. Only used to obtain labels in initialise_state().
  for (t in c(21)) {
    data <- data %>% group_by(symbol) %>% mutate(!!sprintf("r%02d_fwd", t) := lead(.data[[sprintf("r%02d", t)]], n = t)) %>% ungroup()
  }
  
  return(data)
}

initialise_state <- function(data){
  symbols_selected <- sort(head(data$symbol, length(unique(data$symbol))))
  num_symbols <- length(symbols_selected)
  
  unique_dates <- sort(as.Date(unique(data$date)))
  
  # extract data for the 504 + 21 most recent days
  dates_recent <- tail(unique_dates, train_length + lookahead)
  data <- get_features(data)
  data <- data[data$date %in% dates_recent, ]
  day_idx <- 0
  
  # for convenience of processing, we will convert df_recent to a list indexed by dates
  df_list <- split(data, data$date)
  df_list <- lapply(df_list, function(sub_df){
    tmp <- matrix(NA, num_symbols, ncol(data %>% select(-symbol, -date))) # 32 col
    rownames(tmp) <- symbols_selected
    colnames(tmp) <- colnames(sub_df)[-c(1,2)]
    tmp[sub_df$symbol, ] <- as.matrix(sub_df[, -c(1,2)])
    tmp
  })

  # compute the `r21_fwd` column using historical data
  for (i in 1:train_length){
    df_list[[i]][ ,'r21_fwd'] <- df_list[[i+21]][, 'r21']

  }
  
  # positions is a matrix storing positions of each of the 50 stocks established in the past 21 days, which is initialise to 0.
  positions <- matrix(0, lookahead, num_symbols)
  colnames(positions) <- symbols_selected
  
  # model stores the current best prediction model, initialise to NULL
  model <- NULL
  state <- list(day_idx=day_idx ,positions=positions, df_recent=df_list, model=model)
  return(state)
}

# Function to execute the trading algorithm
trading_algorithm <- function(new_data, state){
  symbols_selected <- sort(head(new_data$symbol, length(unique(new_data$symbol))))
  num_symbols <- length(symbols_selected)
  
  bunch(day_idx, positions, df_recent, model) %=% state
  
  # increment day index by 1
  day_idx <- day_idx + 1

  # drop oldest date 
  df_recent[[1]] <- NULL 
  
  # convert new data to a matrix and append it in df_recent
  
  tmp <- matrix(NA, num_symbols, ncol(state$df_recent[1][[1]]))
  rownames(tmp) <- symbols_selected
  
  colnames(tmp) <- colnames(state$df_recent[1][[1]])
  
  # Compute lag-related indicators for new data using past 50 days data
  # Using the past 50 days (instead of just 26) for each new day's calculation to accuracy and stability
  past_26_dfs <- lapply(
    (length(state$df_recent) - 49):length(state$df_recent),
    function(i) {
      df <- as.data.frame(state$df_recent[[i]])  
      df$symbol <- rownames(df)                 
      df$date <- names(state$df_recent[i])[1]
      df
    }
  )
  past_26_dfs = do.call(rbind, past_26_dfs)
  new_data_features = past_26_dfs %>% get_features() %>% filter(date==max(date))
  tmp[new_data$symbol, c(colnames(tmp))] <- as.matrix(new_data_features[, c(colnames(tmp))])
  
  new_date <- as.character(new_data$date[1]) # get current date
  df_recent[[new_date]] <- tmp
  
  # update the `r21_fwd` column for the 21-day ago matrix
  df_recent[[train_length]][, 'r21_fwd'] <- df_recent[[new_date]][, 'r21']
  
  # train a new model every 63 days
  if (day_idx %% valid_length == 1){
    train_mx <- do.call(rbind, df_recent[1:train_length])
    dtrain <- lgb.Dataset(data=train_mx[, covariates], 
                          label=train_mx[, 'r21_fwd'],
                          categorical_feature=cat_var)
    model <- lgb.train(params = params, data = dtrain, verbose=-1)
  }
  
  # use the model to predict returns
  preds <- predict(model, df_recent[[new_date]][, covariates])
  
  # record the last open position to close
  position_to_close <- positions[lookahead, ]
  
  # move all previous positions down by one row
  positions[2:lookahead, ] <- positions[1:(lookahead-1), ]
  
  # New position: long top 40 ETFs and short bottom 10 ETFs in terms of predicted returns, store in the first row of positions
  bottom_n_stock <- 10
  top_n_stock <- 40
  bottom_n_position <- -2/200
  top_n_position <- -(bottom_n_position * bottom_n_stock / top_n_stock)
  
  is_short <- rank(preds) <= bottom_n_stock
  is_long <- rank(preds) > top_n_stock
  positions[1, ] <- 0
  positions[1, is_short] <- bottom_n_position
  positions[1, is_long] <- top_n_position

  # compute trades needed to establish new combined position
  trades <- positions[1, ] - position_to_close
  
  new_state <- list(day_idx=day_idx, positions=positions, df_recent=df_recent, model=model)
  return(list(trades=trades, new_state=new_state))
}


