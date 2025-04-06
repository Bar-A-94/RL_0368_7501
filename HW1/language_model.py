import numpy as np
word_size = 5
b_after_letter = [0.1, 0.4, 0.2, 1] # B, K, O, -
k_after_letter = [0.325, 0, 0.2, 0] # B, K, O, -
o_after_letter = [0.25, 0.4, 0.2, 0] # B, K, O, -
end_after_letter = [0.325, 0.2, 0.4, 0] # B, K, O, -

matrix = [b_after_letter, k_after_letter, o_after_letter, end_after_letter]
p = np.zeros((word_size,3))
q = np.zeros((word_size,3))
p[0,0] = np.log(1)
p[0,1] = -np.inf
p[0,2] = -np.inf

for i in range (1, word_size):
    for j in range(3):
        # Calculate the values for all k first
        values = np.array([p[i-1, k] + np.log(matrix[j][k]) for k in range(3)])
        p[i, j] = np.max(values)
        q[i, j] = np.argmax(values)

# Calculate the last values
for k in range(3):
   print(p[word_size-2, k],matrix[3][k],np.log(matrix[3][k]))
last_values = np.array([p[word_size-1, k] + np.log(matrix[3][k]) for k in range(3)])
last_p = np.max(last_values)
last_q = np.argmax(last_values)
backtrack = [last_q]
for i in range(word_size-1,0,-1):
    backtrack.append(int(q[i, backtrack[-1]]))

backtrack.reverse()
letter_map = {0: 'B', 1: 'K', 2: 'O'}
word = [letter_map[i] for i in backtrack]
word = ''.join(word)

print("Debug information:")
print("p matrix (log probabilities):")
print(p)
print("\np matrix (probabilities):")
print(np.exp(p))
print("\nq matrix:")
print(q)
print("\nlast_p (log probability):", last_p)
print("last_p (probability):", np.exp(last_p))
print("$$$$$$ Final answer $$$$$$")
print("Backtrack:", backtrack)
print("Word:", word)
