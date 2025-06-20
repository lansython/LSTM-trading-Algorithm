library(dplyr)
library(lubridate)
library(TTR)
library(zoo)
library(keras)
library(putils)
library(tensorflow)
library(zeallot)


#######################################################
#Feature Engineering for training  data
#######################################################
Feature_Engineering<- function(data){
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
  # returns
  lags <- c(1, 5, 10, 21, 42, 63)
  for (lag in lags) { # returns
    data <- data %>% mutate(!!paste0("r", lag) := (close / lag(close, lag) - 1))
  }
  # Compute Lag returns deciles
  for (t in T) {
    data = data %>% group_by(date) %>%mutate(!!sprintf("r%ddec", t) := ntile(.data[[sprintf("r%d", t)]], 10)) %>% ungroup()
  }
  data <- data %>%  # Extract time index components
    mutate(year = year(date), month = month(date), weekday = wday(date, week_start = 1))
  forward_lags <- c(1, 5, 21)
  for (lag in forward_lags) {
    data <- data %>%
      mutate(!!paste0(lag, "r_fwd") := (lead(close, lag) / close) - 1)
  }
  return(data)
}

normalize_features <- function(df, covariates) {
  for (co in covariates) {
    col_vals <- df[[co]]
    if (all(is.na(col_vals))) {
      df[[co]] <- NA
    } else {
      rng <- range(col_vals, na.rm = TRUE)
      denom <- diff(rng)
      if (denom == 0) {
        df[[co]] <- 0
      } else {
        df[[co]] <- (col_vals - rng[1]) / denom
      }
    }
  }
  return(df)
}


############################################################
# Generate the sequences to use LSTM
###########################################################
generate_sequences <- function(df, window_size, response_vars, response_var, covariates,  features, other_features) {
  tickers <- sort(unique(df$symbol))
  X_list <- list()
  y_list <- list()
  covar_list <- list()
  symbols_list <- list()
  
  for (ticker in tickers) {
    sub_df <- df[df$symbol == ticker,]
    num_seq <- nrow(sub_df) - window_size + 1
    if (num_seq <= 0) next
    # X is a order 3 tensor
    num_seq <- nrow(sub_df) - window_size + 1 # number of usable sequences
    X_ticker <- array(dim = c(num_seq, window_size, length(features)))
    y_ticker <- numeric(num_seq)
    covar_ticker <- matrix(nrow = num_seq, ncol = length(other_features))
  
    for (i in 1:num_seq) {
      X_ticker[i, , ] <- as.matrix(sub_df[i:(i + window_size - 1), features])
      y_ticker[i] <- sub_df[i + window_size - 1, response_var]
      covar_ticker[i, ] <- as.matrix(sub_df[i + window_size - 1, other_features])
    }
    X_list <- append(X_list, list(X_ticker))
    y_list <- append(y_list, list(y_ticker))
    covar_list <- append(covar_list, list(covar_ticker))
    symbols_list <- append(symbols_list, rep(ticker, num_seq))
  }
  X <- do.call(abind::abind, list(X_list, along=1))
  y <- unlist(y_list)
  covar <- do.call(rbind, covar_list)
  symbols <- unlist(symbols_list)
  symbols <- matrix(as.integer(factor(symbols, levels=tickers)) - 1, ncol=1)
  return(list(X = X, y = y, covar = covar, symbols = symbols))
}


#######################################################
#Keras Model
#######################################################
Building_keras<- function(lstm1_units, lstm2_units, embedding_dim, window_size, num_symbols, features, other_features){
  # Define input layers
  price_input <- layer_input(shape = c(window_size, length(features)), name = 'price_input')
  symbol_input <- layer_input(shape = c(1), name = 'symbol_input')
  other_input <- layer_input(shape = c(length(other_features)), name='other_input')
  
  # Embedding layers
  symbol_embed <- symbol_input %>%
    layer_embedding(input_dim = num_symbols, output_dim = embedding_dim) %>%
    layer_flatten()
  
  # stacked LSTM layers
  lstm_out <- price_input %>%
    layer_lstm(units = lstm1_units, return_sequences = TRUE, dropout = 0.2) %>%
    layer_lstm(units = lstm2_units, dropout = 0.2)
  
  # Combine inputs
  merged <- layer_concatenate(list(lstm_out, symbol_embed, other_input))
  
  # Fully connected layers
  bn <- layer_batch_normalization(merged)
  hidden <- layer_dense(bn, units = 10, activation = "relu")
  output <- layer_dense(hidden, units = 1, activation = "linear")
  model <- keras_model(inputs = list(price_input, symbol_input, other_input), 
                       outputs = output)
  return(model)
}





initialise_state<- function(data){
  
  ################## Find optimal tainign size and then keep that trtainign data + tuning needed 
  #Feature Engineering
  data<-Feature_Engineering(data)
  
  #Preparation for LSTM
  response_vars <- c('1r_fwd', "5r_fwd", "21r_fwd")
  response_var <- "1r_fwd"
  data<- data <- data[1:(nrow(data) - 5), ]# remove Na in response columns
  covariates <- setdiff(names(data), c(response_vars, 'date', 'symbol',"year","weekday", "month"))
  features <- c("open", "close", "low", "high", "volume")
  other_features <- setdiff(covariates, features)
  data <- normalize_features(data, covariates)
  
  window_size <- 63
  bunch(X_train, y_train, covar_train, symbols_train) %=% generate_sequences(data, window_size, response_vars, response_var, covariates,  features, other_features)
  
  #Keras
  num_symbols <- length(unique(data$symbol))
  lstm1_units <- 50
  lstm2_units <- 10
  embedding_dim <- 5
  model<- Building_keras(lstm1_units, lstm2_units, embedding_dim, window_size, num_symbols, features, other_features)
  compile(model, optimizer = 'rmsprop', loss = 'mse')
  history <- fit(model, x = list(X_train, symbols_train, covar_train), y = y_train,
                 epochs=7, batch_size=64)
  
  
  #Initialize state
  symbols_current <- sort(unique(data$symbol))
  positions <- matrix(0, 1, length(symbols_current))
  colnames(positions) <- sort(symbols_current)
  day_idx <- 0
  unique_dates <- sort(unique(data$date))
  dates_recent <- tail(unique_dates, window_size)
  data_initial <- data[data$date %in% dates_recent, ]
  state <- list(day_idx= day_idx, positions=positions, 
                df_recent=data_initial, model=model, lookahead=1, covariates = covariates, features = features, other_features = other_features, window_size = window_size)
  str(state)
  length(state)
  names(state)
  return(state)
}



trading_algorithm <- function(new_data, state){
  
  bunch(day_idx, positions, df_recent, model, lookahead, covariates , features , other_features, window_size) %=% state
  
  response_var <- "1r_fwd"
  # increment day index by 1
  day_idx <- day_idx + 1
  df_recent <- bind_rows(df_recent, new_data)
  # Only keep the window sizew rows per symbol (efficient!)
  max_needed_rows <- 70 
  df_recent <- df_recent %>%
    group_by(symbol) %>%
    arrange(date) %>%
    slice_tail(n = max_needed_rows) %>%
    ungroup()
  
  # Now engineer features on this trimmed data
  df_recent <- Feature_Engineering(df_recent)
  df_recent <- normalize_features(df_recent, covariates)
  
  new_data <- tail(df_recent, 1)
  
  
  num_features <- ncol(df_recent) - 2
  symbols_current <- sort(unique(df_recent$symbol))  
  df_dict <- lapply(split(df_recent, df_recent$date), function(sub_df){ #Used as df_list might not always be the same dimension in case a stock is not traded on a specific day 
    tmp <- matrix(NA, length(symbols_current), num_features)
    rownames(tmp) <- symbols_current
    colnames(tmp) <- colnames(sub_df)[-c(1,2)]
    tmp[sub_df$symbol, ] <- as.matrix(sub_df[, -c(1,2)])
    tmp
  })

  
  # drop oldest date 
  df_dict[[1]] <- NULL
  # convert new data to a matrix and append it in df_recent
  tmp <- matrix(NA, length(symbols_current), num_features)
  rownames(tmp) <- symbols_current
  colnames(tmp) <- colnames(new_data)[-c(1,2)]
  tmp[new_data$symbol, ] <- as.matrix(new_data[, -c(1,2)])
  new_date <- as.character(new_data$date[1]) # get current date
  df_dict[[new_date]] <- tmp
  
  train_length<- length(df_dict)
  df_dict[[train_length]][, '1r_fwd'] <- df_dict[[new_date]][, 'r1']
  
  if (day_idx %% 63 == 1){
    
    # Prepare training data from df_dict
    df_train <- do.call(rbind, tail(df_dict, train_length))  # flatten last N matrices
    df_train <- as.data.frame(df_train)
    df_train$symbol <- rep(symbols_current, train_length)
    df_train$date <- rep(names(tail(df_dict, train_length)), each = length(symbols_current))
    
    response_var <- "1r_fwd"
    window_size <- 63
    bunch(X_train, y_train, covar_train, symbols_train) %=% generate_sequences(df_train, window_size, response_vars, response_var, covariates,  features, other_features)
    
    # Build and train model
    model <- Building_keras(
      lstm1_units = 50,
      lstm2_units = 10,
      embedding_dim = 5,
      window_size = window_size,
      num_symbols = length(symbols_current),
      features = features,
      other_features = other_features
    )
    
    compile(model, optimizer = 'rmsprop', loss = 'mse')
    
    fit(model,
        x = list(X_train, symbols_train, covar_train),
        y = y_train,
        epochs = 7,
        batch_size = 64,
        verbose = 0)
  }

  # Instead of feeding entire df_test to generate_sequences,
  # extract just the last 63 days per symbol
  latest_data <- df_recent %>%
    group_by(symbol) %>%
    arrange(date) %>%
    slice_tail(n = window_size) %>%
    ungroup()
  
  # Now pass only this into sequence generator
  bunch(X_test, y, covar_test, symbols_test) %=% generate_sequences(
    latest_data, window_size, response_vars, response_var, covariates, features, other_features
  )
  preds <- predict(model, list(X_test, symbols_test, covar_test), verbose = 0)
  
  #update position
  position_to_close <- positions[lookahead, ]
  # Only shift rows if lookahead > 1
  if (lookahead > 1) {
    positions[2:lookahead, ] <- positions[1:(lookahead - 1), ]
  }
  
  # Long top 50, short bottom 50
  is_short <- rank(preds) <= 50
  is_long <- rank(preds) > (length(preds) - 50)
  positions[1, ] <- 0
  positions[1, is_short] <- -1/100
  positions[1, is_long] <- 1/100
  
  
  
  #compute trades
  trades <- positions[1, ] - position_to_close
  
  new_state <- list(day_idx = day_idx,
                    positions = positions,
                    df_recent = df_recent,
                    model = model,
                    lookahead = lookahead,
                    covariates = covariates,
                    features = features,
                    other_features = other_features,
                    window_size = window_size)
  
  return(list(trades = trades, new_state = new_state))
}









