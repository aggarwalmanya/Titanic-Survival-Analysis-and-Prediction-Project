# Install required packages if not already installed
install.packages("ggplot2")
install.packages("dplyr")
install.packages("caret")
install.packages("readr")

# Load the libraries
library(ggplot2)
library(dplyr)
library(caret)
library(readr)

# Load the Titanic dataset
titanic_url <- "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data <- read_csv(titanic_url, col_names = TRUE, col_types = cols())

# Display the first few rows of the dataset
head(titanic_data)

# Summary statistics of the dataset
summary(titanic_data)

# Check for missing values
colSums(is.na(titanic_data))

# Fill missing values for Age with the median age
titanic_data$Age[is.na(titanic_data$Age)] <- median(titanic_data$Age, na.rm = TRUE)

# Fill missing values for Embarked with the mode
titanic_data$Embarked[is.na(titanic_data$Embarked)] <- as.character(names(sort(table(titanic_data$Embarked), decreasing = TRUE))[1])

# Convert categorical variables to factors
categorical_vars <- c("Sex", "Embarked", "Pclass", "Survived")
titanic_data[categorical_vars] <- lapply(titanic_data[categorical_vars], as.factor)

# Create a new feature for family size
titanic_data <- titanic_data %>%
  mutate(FamilySize = SibSp + Parch + 1)

# Summary of the cleaned dataset
summary(titanic_data)

# Plot relationship between passenger class and survival
ggplot(titanic_data, aes(x = Pclass, fill = Survived)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Passenger Class vs Survival", x = "Passenger Class", y = "Proportion Survived")

# Plot relationship between sex and survival
ggplot(titanic_data, aes(x = Sex, fill = Survived)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Sex vs Survival", x = "Sex", y = "Proportion Survived")

# Plot relationship between age and survival
ggplot(titanic_data, aes(x = Age, fill = Survived)) +
  geom_histogram(bins = 30, position = "fill") +
  theme_minimal() +
  labs(title = "Age vs Survival", x = "Age", y = "Proportion Survived")

# Plot relationship between family size and survival
ggplot(titanic_data, aes(x = FamilySize, fill = Survived)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Family Size vs Survival", x = "Family Size", y = "Proportion Survived")

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(titanic_data$Survived, p = 0.8, list = FALSE)
train_data <- titanic_data[train_index, ]
test_data <- titanic_data[-train_index, ]

# Build a logistic regression model
model <- glm(Survived ~ Pclass + Sex + Age + FamilySize, data = train_data, family = binomial)

# Print the model summary
summary(model)

# Predict on the test data
predictions <- predict(model, test_data, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Calculate accuracy
accuracy <- mean(predicted_classes == test_data$Survived)
print(paste("Accuracy:", accuracy))
