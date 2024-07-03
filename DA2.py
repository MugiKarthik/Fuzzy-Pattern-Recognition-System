import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Define linguistic variables
eyes = ['Closed', 'Open']
mouth = ['Smiling', 'Frowning', 'Neutral']
mood = ['Happy', 'Sad', 'Angry', 'Neutral']

# Define fuzzy membership functions for each variable
eyes_mf = dict(Closed=fuzz.trimf(np.arange(0, 6), [0, 0, 3]),
               Open=fuzz.trimf(np.arange(0, 6), [2, 5, 5]))
mouth_mf = dict(Smiling=fuzz.trimf(np.arange(0, 6), [0, 0, 3]),
                Frowning=fuzz.trimf(np.arange(0, 6), [2, 5, 5]),
                Neutral=fuzz.trimf(np.arange(0, 6), [2, 2, 5]))
mood_mf = dict(Happy=fuzz.trimf(np.arange(0, 6), [0, 0, 3]),
               Sad=fuzz.trimf(np.arange(0, 6), [2, 5, 5]),
               Angry=fuzz.trimf(np.arange(0, 6), [2, 2, 5]),
               Neutral=fuzz.trimf(np.arange(0, 6), [4, 5, 5]))

# Plot fuzzy membership functions
fig, ax = plt.subplots(nrows=3, figsize=(8, 9))

# Plot eyes membership functions
for label, membership in eyes_mf.items():
    ax[0].plot(np.arange(0, 6), membership, label=label)
ax[0].set_title('Eyes Membership Functions')
ax[0].set_xlabel('Eyes Value')
ax[0].set_ylabel('Membership Degree')
ax[0].legend()

# Plot mouth membership functions
for label, membership in mouth_mf.items():
    ax[1].plot(np.arange(0, 6), membership, label=label)
ax[1].set_title('Mouth Membership Functions')
ax[1].set_xlabel('Mouth Value')
ax[1].set_ylabel('Membership Degree')
ax[1].legend()

# Plot mood membership functions
for label, membership in mood_mf.items():
    ax[2].plot(np.arange(0, 6), membership, label=label)
ax[2].set_title('Mood Membership Functions')
ax[2].set_xlabel('Mood Value')
ax[2].set_ylabel('Membership Degree')
ax[2].legend()

# Define rules
rules = [
    ('Closed', 'Smiling', 'Happy'),
    ('Closed', 'Frowning', 'Sad'),
    ('Open', 'Smiling', 'Happy'),
    ('Open', 'Frowning', 'Sad'),
    ('Open', 'Neutral', 'Neutral')
]

def fuzzy_inference(eyes_val, mouth_val):
    result = np.zeros(len(mood))
    for rule in rules:
        eyes_mf_val = fuzz.interp_membership(np.arange(0, 6), eyes_mf[rule[0]], eyes_val)
        mouth_mf_val = fuzz.interp_membership(np.arange(0, 6), mouth_mf[rule[1]], mouth_val)
        activation_strength = np.fmin(eyes_mf_val, mouth_mf_val)
        result[mood.index(rule[2])] = max(result[mood.index(rule[2])], activation_strength)
    return result


# Example
emoji = ":)"
eyes_val = eyes.index('Open')
mouth_val = mouth.index('Frowning')
output_mood = fuzzy_inference(eyes_val, mouth_val)
predicted_mood_index = np.argmax(output_mood)
predicted_mood = mood[predicted_mood_index]
defuzz_value = fuzz.defuzz(np.arange(len(mood)), output_mood, 'centroid')

# Highlight defuzzification value on the mood membership plot
ax[2].axvline(x=defuzz_value, color='r', linestyle='--', label=f'Defuzzification Value: {defuzz_value:.2f}')
ax[2].legend()

plt.tight_layout()
plt.show()

print("Predicted Mood:", predicted_mood)
