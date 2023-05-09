library(tidytext)
library(tidyverse)

afinn <- get_sentiments("afinn")

surprise <- afinn %>%
  filter(word == "surprise") # missing, made up of joy and fear and positive, avg = 1
happy <- afinn %>%
  filter(word == "happy") # 3
disgust <- afinn %>%
  filter(word == "disgust") # -3
fear <- afinn %>%
  filter(word == "fear") # -2, also for surprise
neutral <- afinn %>%
  filter(word == "neutral") # missing, made up of anticipation and trust in nrc, avg = 1
angry <- afinn %>%
  filter(word == "angry") # -3
sad <- afinn %>%
  filter(word == "sad") # -2
anticipation <- afinn %>%
  filter(word == "anticipation") # 1, for neutral
trust <- afinn %>%
  filter(word == "trust") # 1, for neutral
joy <- afinn %>%
  filter(word == "joy") # 3, for surprise
positive <- afinn %>%
  filter(word == "positive") # 2, for surprise

bing <- get_sentiments("bing")

surprise <- bing %>%
  filter(word == "surprise") # missing
happy <- bing %>%
  filter(word == "happy") # 3, positive
disgust <- bing %>%
  filter(word == "disgust") # -3, negative
fear <- bing %>%
  filter(word == "fear") # -2, negative
neutral <- bing %>%
  filter(word == "neutral") # missing
angry <- bing %>%
  filter(word == "angry") # -3, negative
sad <- bing %>%
  filter(word == "sad") # -2, negative

nrc <- get_sentiments("nrc")

surprise <- nrc %>%
  filter(word == "surprise") # missing
happy <- nrc %>%
  filter(word == "happy") # 3, positive
disgust <- nrc %>%
  filter(word == "disgust") # -3, negative
fear <- nrc %>%
  filter(word == "fear") # -2, negative
neutral <- nrc %>%
  filter(word == "neutral") # missing
angry <- nrc %>%
  filter(word == "angry") # -3, negative
sad <- nrc %>%
  filter(word == "sad") # -2, negative

loughran <- get_sentiments("loughran")

surprise <- loughran %>%
  filter(word == "surprise") # missing
