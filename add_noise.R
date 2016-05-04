# this script adds noise to a .csv file
input_file <-  "flight_data.csv"
output_file <- "measured_data.csv"
sigma_accel <- 2
sigma_alt <- 10
add_noise <- function(input_file, output_file, sigma_accel, sigma_alt){
  data <- read.csv(input_file, header = TRUE)
  num_obs <- nrow(data)
  true_accel <- data$Vertical.acceleration..m.s..
  true_alt <- data$Altitude..m.
  time <- data$X..Time..s.
  noisy_accel <- true_accel + rnorm(num_obs, 0, sigma_accel^2)
  noisy_alt <- true_alt + rnorm(num_obs, 0, sigma_alt^2)
  measured_data <- cbind(Time = time, accel = noisy_accel, alt = noisy_alt)
  write.csv(measured_data, output_file, row.names = FALSE)
}