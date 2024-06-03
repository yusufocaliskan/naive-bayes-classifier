
# P(S_i | x) * P(x) = P(x | S_i) * P(S_i)
def bayes_theorem(ls,lx,rx,rs):
    l_result = ls*lx
    r_result = rx*rs
    return l_result, r_result

l_s =0.9 
l_x =0.2 
r_x = 0.5
r_s = 0.2

l_result,r_result = bayes_theorem(l_s,l_x, r_x, r_s)
print("Result(bayes_theorem):", l_result,r_result)


# classification
# one dimensional
# p(x)=∑ i=1L p(x∣Si)P(Si)
def total_probability(x, s):

    px = 0
    for i in range(len(s)):
        px += x[i] * s[i]
    
    return px


# "Spam" and "Not Spam"
sVal  = [0.4,0.5] # %40, %50 

# Probilities of `free` word that the both can contain
# Free in 'Spam' emails 
# Free in 'Not Spam' emails 
xVal  = [0.5,0.2] 

result = total_probability(xVal, sVal)
print("Result(total_probability):", result)


# Most probable
#P(x|S_{i})P(S_{i})>P(x|S_{j})P(S_{j})}, ∀j ≠,𝑖
def bayes_decision_theorem(pxs, ps):
    
    max_probability = 0
    best_class = -1
    for i in range(len(ps)):
        probability = pxs[i] * ps[i]
        if  probability > max_probability: 
            max_probability = probability
            best_class = i
    return best_class

pxs = [0.8,0.1]
ps = [0.4,0.6]

result = bayes_decision_theorem(pxs, ps)
print("Result(bayes_decision_theorem):", result)

def naive_bayes_classify(pxs, ps):
    max_prob = -1
    best_class = -1

    for i in range(len(ps)):
        prob = ps[i]
        for j in range(len(pxs[0])):
            pxj = pxs[i][j]
            prob *= pxj 
            if  prob > max_prob:
                max_prob = prob
                best_class = i

    return best_class, max_prob


# Rainy day, Rainless day 
#Classes
ps = [0.4, 0.6] 

# Cloudy, Windy, Chilly 
# Features

pxs =[  
    [0.8, 0.6, 0.7],  
    [0.2, 0.4, 0.5]
] 

bes_class, max_prob = naive_bayes_classify(pxs, ps)
print("Result(naive_bayes_classify):", bes_class, max_prob)

