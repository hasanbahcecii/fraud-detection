"""
Fraud Detection With Transformers - Synthetic Data Generation
"""

import numpy as np
import pandas as pd

# define seed for reproducibility
np.random.seed(42)

# each sample includes 20 actions
sequence_length = 20

# number of features: click_duration, page_duration, amount_spent, scroll_depth, cart_events
n_features = 5

# number of samples for each class
n_samples_per_class = 500

total_samples = n_samples_per_class * 3

# empty list for each class
data = []
labels = []

# Class 0: Normal User
for _ in range(n_samples_per_class):
    # Normal user behavior: moderate clicks, reasonable time on page, average spending, typical scrolling, occasional cart events
    click_count = np.random.normal(loc=3, scale=1, size=sequence_length)
    page_duration = np.random.normal(loc=5, scale=2, size=sequence_length)
    amount_spent = np.random.normal(loc=10, scale=5, size=sequence_length)
    scroll_depth = np.random.normal(loc=50, scale=15, size=sequence_length)
    cart_events = np.random.poisson(lam=0.5, size=sequence_length)

    sample = np.stack([click_count, page_duration, amount_spent, scroll_depth, cart_events], axis=1)
    data.append(sample)
    labels.append(0)  # label: normal user


# Class 1: Bot
for _ in range(n_samples_per_class):
    # Bot behavior: very high click frequency, very short page duration (not reading content),
    # low or random spending, shallow scrolling, rare cart events
    click_count = np.random.normal(loc=10, scale=3, size=sequence_length)   # excessive clicking
    page_duration = np.random.normal(loc=1, scale=0.5, size=sequence_length)  # almost no time spent
    amount_spent = np.zeros(size=sequence_length)   # no spending
    scroll_depth = np.random.normal(loc=10, scale=5, size=sequence_length)  # barely scrolling
    cart_events = np.zeros(size=sequence_length)          # no cart activity

    sample = np.stack([click_count, page_duration, amount_spent, scroll_depth, cart_events], axis=1)
    data.append(sample)
    labels.append(1)  # label: bot


# Class 2: Fraudster
for _ in range(n_samples_per_class):
    # Fraudster behavior: moderate clicks, normal page duration (to look legitimate),
    # unusually high spending, deep scrolling (to mimic real users), abnormal cart events (adding/removing frequently)
    click_count = np.random.normal(loc=4, scale=4, size=sequence_length)    # slightly higher than normal
    page_duration = np.random.normal(loc=6, scale=3, size=sequence_length) # looks normal
    amount_spent = np.random.exponential(scale=20, size=sequence_length) # suspiciously high spending
    scroll_depth = np.random.normal(loc=80, scale=20, size=sequence_length) # deeper scrolling
    cart_events = np.random.poisson(lam=2, size=sequence_length)            # frequent cart manipulation

    sample = np.stack([click_count, page_duration, amount_spent, scroll_depth, cart_events], axis=1)
    data.append(sample)
    labels.append(2)  # label: fraudster


# data: (1500, 20, 5) = (num of samples, num of steps, num of features)
# labels: (1500, 1)
data = np.array(data)
labels = np.array(labels)

# shuffle the data
indices = np.arange(total_samples)
np.random.shuffle(indices)
data = data[indices]
labels = data[indices]

# save data
np.save("X_fraud.npy", data)
np.save("y_fraud.npy", labels)

# alternatively save as csv
df = pd.DataFrame(data[0], columns= [])