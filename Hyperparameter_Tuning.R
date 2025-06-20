################################################################
#Time series split for Tuning
################################################################
time_series_split <- function(dates, n_splits=0, 
                              train_length=126, valid_length=21, lookahead=1){
  unique_dates <- sort(unique(dates), decreasing=TRUE)
  
  if (n_splits == 0){
    block_length <- train_length + valid_length + lookahead
    n_splits <- (length(unique_dates) - block_length) %/% valid_length
  }
  
  valid_end_idx <- ((1:n_splits) - 1) * valid_length + 1
  valid_start_idx <- valid_end_idx + valid_length - 1
  train_end_idx <- valid_start_idx + lookahead + 1
  train_start_idx <- train_end_idx + train_length - 1
  
  splits <- list()
  for (i in 1:n_splits){
    train = which(dates >= unique_dates[train_start_idx[i]] & 
                    dates <= unique_dates[train_end_idx[i]])
    test = which(dates >= unique_dates[valid_start_idx[i]] & 
                   dates <= unique_dates[valid_end_idx[i]])
    splits[[i]] <- list(train = train, test = test)
  }
  
  return(splits)
}


if (FALSE){
  ###################################################
  # Hyperparameter Tuning
  ###################################################
  response_vars <- c('1r_fwd', "5r_fwd", "21r_fwd")
  response_var <- "5r_fwd"
  data<- data <- data[1:(nrow(data) - 5), ]# remove Na in response columns
  covariates <- setdiff(names(data), c(response_vars, 'date', 'symbol',"year","weekday", "month"))
  features <- c("open", "close", "low", "high", "volume")
  other_features <- setdiff(covariates, features)
  
  
  compute_ic <- function(dates, y_true, y_pred) {
    df <- data.frame(date = dates, y_true = y_true, y_pred = y_pred)
    ic_by_day <- df %>%
      group_by(date) %>%
      summarise(ic = cor(y_true, y_pred, method = "spearman", use = "complete.obs")) %>%
      pull(ic)
    mean(ic_by_day, na.rm = TRUE)
  }
  
  
  param_df <- expand.grid(
    train_length = c(252, 252*2),
    valid_length = c(21, 63),
    lookahead = c(1, 5, 21),
    lstm1_units = c(10, 20, 50),
    lstm2_units = c(10, 20),
    embedding_dim = c(3, 5),
    window_size = c(21,42, 63),
    batch_size = c(32, 64),
    epochs = c(5, 10),
    stringsAsFactors = FALSE
  )
  
  set.seed(458)
  num_param_comb <- 50
  param_df <- param_df[sample(nrow(param_df), num_param_comb), ]
  param_df$ic <- 0
  
  for (i in 1:num_param_comb) {
    bunch(train_length, valid_length, lookahead,
          lstm1_units, lstm2_units, embedding_dim, window_size,
          batch_size, epochs) %=% param_df[i, 1:9]
    
    response_var <- paste0(lookahead, "r_fwd")
    response_vars <- c(response_var)
    ic_by_day <- c()
    
    splits <- time_series_split(data$date,
                                train_length = train_length,
                                valid_length = valid_length,
                                lookahead = lookahead)
    
    for (fold in 1:length(splits)) {
      bunch(train_idx, valid_idx) %=% splits[[fold]]
      
      train_data <- data[train_idx, ]
      valid_data <- data[valid_idx, ]
      
      # 🚨 Feature engineering and normalization on full data (potential leakage)
      full_data <- bind_rows(train_data, valid_data)
      full_data <- Feature_Engineering(full_data)
      full_data <- normalize_features(full_data, covariates)
      
      train_data <- full_data[1:nrow(train_data), ]
      valid_data <- full_data[(nrow(train_data)+1):nrow(full_data), ]
      
      
      # Generate sequences directly from each
      bunch(X_train, y_train, covar_train, symbols_train) %=% generate_sequences(
        train_data, window_size, response_vars, response_var, covariates, features, other_features
      )
      bunch(X_valid, y_valid, covar_valid, symbols_valid) %=% generate_sequences(
        valid_data, window_size, response_vars, response_var, covariates, features, other_features
      )
      if (is.null(X_valid) || is.null(y_valid) || is.null(covar_valid) || is.null(symbols_valid)) {
        cat(sprintf("⚠️  Skipping fold %d in param set %d due to NULL validation input.\n", fold, i))
        next
      }
      
      num_symbols <- length(unique(train_data$symbol))
      model <- Building_keras(lstm1_units, lstm2_units, embedding_dim, window_size, num_symbols, features, other_features)
      compile(model, optimizer = 'rmsprop', loss = 'mse')
      
      fit(model, x = list(X_train, symbols_train, covar_train),
          y = y_train,
          epochs = epochs,
          batch_size = batch_size,
          verbose = 0)
      
      preds <- predict(model, list(X_valid, symbols_valid, covar_valid), verbose = 0)
      
      valid_dates_for_ic <- tail(valid_data$date, length(y_valid))
      ic_fold <- compute_ic(valid_dates_for_ic, y_valid, preds)
      ic_by_day <- c(ic_by_day, ic_fold)
    }
    
    param_df$ic[i] <- mean(ic_by_day)
    print(sprintf("Finished %d / %d | IC = %.4f", i, num_param_comb, param_df$ic[i]))
  }
  
  i_opt <- which.max(param_df$ic)
  best_params <- param_df[i_opt, ]
  print(best_params)
}