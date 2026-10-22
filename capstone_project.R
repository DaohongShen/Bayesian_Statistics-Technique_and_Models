# Load the dataset
data("EuStockMarkets")

# View dataset documentation
?EuStockMarkets

# Inspect the data
head(EuStockMarkets)

# Check dimensions
dim(EuStockMarkets)

# Check missing values
colSums(is.na(EuStockMarkets))

# Convert prices into daily log returns (%)
returns = as.data.frame(
  100 * diff(log(EuStockMarkets))
)

# Inspect returns
head(returns)
summary(returns)

# Correlation between DAX and FTSE
cor(returns$DAX, returns$FTSE)




# Step 3: Exploratory Data Analysis

x = returns$DAX
y = returns$FTSE

par(mfrow = c(2, 2))

# Figure 1: Scatter plot
plot(x, y,
     main = "DAX vs FTSE Daily Returns",
     xlab = "DAX Return (%)",
     ylab = "FTSE Return (%)",
     pch = 16,
     cex = 0.5,
     col = "steelblue")
abline(lm(y ~ x), col = "red", lwd = 2)

# Figure 2: Histogram of FTSE returns
hist(y,
     breaks = 40,
     probability = TRUE,
     main = "Distribution of FTSE Returns",
     xlab = "FTSE Return (%)",
     col = "lightblue")

curve(dnorm(x, mean(y), sd(y)),
      add = TRUE, col = "red", lwd = 2)

# Figure 3: Normal Q-Q plot
qqnorm(y,
       main = "Normal Q-Q Plot: FTSE",
       pch = 16, cex = 0.5)
qqline(y, col = "red", lwd = 2)

# Figure 4: Returns over time
plot(y, type = "l",
     main = "FTSE Daily Returns Over Time",
     xlab = "Observation Number",
     ylab = "FTSE Return (%)",
     col = "steelblue")

par(mfrow = c(1, 1))



library(rjags)
library(coda)

# Step 5: Bayesian Normal Regression

model_string1 = "
model {
  for (i in 1:N) {
    y[i] ~ dnorm(mu[i], tau)
    mu[i] <- alpha + beta * x[i]
  }

  alpha ~ dnorm(0, 1)
  beta ~ dnorm(0, 0.25)

  sigma ~ dunif(0, 10)
  tau <- 1 / (sigma * sigma)
}
"

# Prepare data
data_jags = list(
  N = nrow(returns),
  y = returns$FTSE,
  x = returns$DAX
)

# Fit model
set.seed(123)

model1 = jags.model(
  textConnection(model_string1),
  data = data_jags,
  n.chains = 3,
  n.adapt = 1000
)

# Burn-in
update(model1, 5000)

# Posterior sampling
samples1 = coda.samples(
  model1,
  variable.names = c("alpha", "beta", "sigma"),
  n.iter = 10000
)

# Posterior summary
summary(samples1)



# MCMC convergence
gelman.diag(samples1)

# Effective sample size
effectiveSize(samples1)

# Posterior probability beta > 0
posterior1 = as.matrix(samples1)
mean(posterior1[, "beta"] > 0)

# 95% credible interval
quantile(
  posterior1[, "beta"],
  c(0.025, 0.975)
)




# Step 6: Model Diagnostics

# Extract posterior means
posterior1 = as.matrix(samples1)

alpha_hat = mean(posterior1[, "alpha"])
beta_hat = mean(posterior1[, "beta"])
sigma_hat = mean(posterior1[, "sigma"])

# Calculate fitted values and residuals
fitted1 = alpha_hat + beta_hat * returns$DAX
residuals1 = returns$FTSE - fitted1

# Four diagnostic plots
par(mfrow = c(2, 2))

# 1. Residuals vs fitted values
plot(fitted1, residuals1,
     main = "Residuals vs Fitted",
     xlab = "Fitted Values",
     ylab = "Residuals",
     pch = 16, cex = 0.5,
     col = "steelblue")
abline(h = 0, col = "red", lwd = 2)

# 2. Normal Q-Q plot of residuals
qqnorm(residuals1,
       main = "Normal Q-Q Plot of Residuals",
       pch = 16, cex = 0.5)
qqline(residuals1, col = "red", lwd = 2)

# 3. Autocorrelation of residuals
acf(residuals1,
    main = "ACF of Residuals")

# 4. Autocorrelation of squared residuals
acf(residuals1^2,
    main = "ACF of Squared Residuals")

par(mfrow = c(1, 1))




# Number of residuals beyond 3 standard deviations
observed_extreme = sum(
  abs(residuals1) > 3 * sigma_hat
)

# Expected number under Normal errors
expected_extreme = length(residuals1) *
  2 * (1 - pnorm(3))

# Residual kurtosis
r = residuals1 - mean(residuals1)

kurtosis1 = mean(r^4) / mean(r^2)^2

cat("Observed extreme residuals:",
    observed_extreme, "\n")

cat("Expected under Normal model:",
    expected_extreme, "\n")

cat("Residual kurtosis:",
    kurtosis1, "\n")



# Step 7: Student-t Bayesian Regression

model_string2 = "
model {
  for (i in 1:N) {
    y[i] ~ dt(mu[i], tau, nu)
    mu[i] <- alpha + beta * x[i]
  }

  alpha ~ dnorm(0, 1)
  beta ~ dnorm(0, 0.25)

  sigma ~ dunif(0, 10)
  tau <- 1 / (sigma * sigma)

  eta ~ dexp(0.1)
  nu <- 2 + eta
}
"

set.seed(123)

model2 = jags.model(
  textConnection(model_string2),
  data = data_jags,
  n.chains = 3,
  n.adapt = 1000
)

# Burn-in
update(model2, 5000)

# Posterior samples
samples2 = coda.samples(
  model2,
  variable.names = c(
    "alpha", "beta", "sigma", "nu"
  ),
  n.iter = 10000
)

# Posterior summary
summary(samples2)

# Convergence diagnostics
gelman.diag(samples2)
effectiveSize(samples2)

# Posterior probability
posterior2 = as.matrix(samples2)
mean(posterior2[, "beta"] > 0)




# DIC comparison

dic1 = dic.samples(
  model1, n.iter = 5000, type = "pD"
)

dic2 = dic.samples(
  model2, n.iter = 5000, type = "pD"
)

DIC1 = sum(dic1$deviance) +
  sum(dic1$penalty)

DIC2 = sum(dic2$deviance) +
  sum(dic2$penalty)

data.frame(
  Model = c("Normal", "Student-t"),
  DIC = c(DIC1, DIC2)
)




# Residual serial correlation
Box.test(
  residuals1,
  lag = 20,
  type = "Ljung-Box"
)

# Volatility clustering
Box.test(
  residuals1^2,
  lag = 20,
  type = "Ljung-Box"
)




# Student-t residuals
alpha2 <- mean(posterior2[, "alpha"])
beta2 <- mean(posterior2[, "beta"])

fitted2 <- alpha2 + beta2 * returns$DAX
residuals2 <- returns$FTSE - fitted2

# Serial correlation
Box.test(residuals2,
         lag = 20,
         type = "Ljung-Box")

# Squared residual autocorrelation
Box.test(residuals2^2,
         lag = 20,
         type = "Ljung-Box")




# Posterior Predictive Check

set.seed(123)

post1 <- as.matrix(samples1)
post2 <- as.matrix(samples2)

N <- nrow(returns)
threshold <- 3 * mean(post1[, "sigma"])

# Observed extreme residuals
observed1 <- sum(abs(residuals1) > threshold)
observed2 <- sum(abs(residuals2) > threshold)

# Sample 500 posterior draws
idx1 <- sample.int(nrow(post1), 500)
idx2 <- sample.int(nrow(post2), 500)

extreme1 <- numeric(500)
extreme2 <- numeric(500)

for (i in 1:500) {
  
  # Normal predictive errors
  e1 <- rnorm(N,
              mean = 0,
              sd = post1[idx1[i], "sigma"])
  
  # Student-t predictive errors
  e2 <- rt(N,
           df = post2[idx2[i], "nu"]) *
    post2[idx2[i], "sigma"]
  
  extreme1[i] <- sum(abs(e1) > threshold)
  extreme2[i] <- sum(abs(e2) > threshold)
}

# Comparison
data.frame(
  Model = c("Observed (Normal fitted mean)",
            "Observed (Student-t fitted mean)",
            "Normal Predictive",
            "Student-t Predictive"),
  Extreme_Count = c(
    observed1,
    observed2,
    mean(extreme1),
    mean(extreme2)
  )
)

# Predictive intervals
quantile(extreme1, c(0.025, 0.5, 0.975))
quantile(extreme2, c(0.025, 0.5, 0.975))
