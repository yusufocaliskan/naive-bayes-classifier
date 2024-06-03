"""

    The Bayes theorem formula 

    P(A|B) =  P(B|A)⋅P(A) / P(B) 
    P(B) = P(B|A)⋅P(A)+P(B|¬A)⋅P(¬A)b

    For more details check this: 
    https://simple.wikipedia.org/wiki/Bayes%27_theorem
"""

def bayes_theorem():


    a = 0.01
    b = 0.99 # In general 
    b_given_a = 0.05 

    a_given_b = (b_given_a * a) / b
    return a_given_b 

print("Result (bayes_theorem) ==> Pro of A: ", bayes_theorem())
    
def calculate_covid_test(p_covid, pos_given_covid, pos_given_not_covid, pos_not_covid):

    """ P(A|B) =  P(B|A)⋅P(A) / P(B) """

    # calculate P(B)
    # P(B)=P(B∣A)⋅P(A)+P(B∣¬A)⋅P(¬A)
    p_pos = (pos_given_covid*p_covid) + (pos_given_not_covid * pos_not_covid)


    # P(A|B) =  P(B|A)⋅P(A) / P(B) 

    p_covid_given_pos = ( pos_given_covid * p_covid) / p_pos 
    return p_covid_given_pos

# 1. probability of the persons who are covid positive
# 2. sensitivity  (posibablity of posive in test) 
# 3. ratio of the negative in test
# 4. ratio of the negative
p_covid = 0.2   
pos_given_covid =0.85   
pos_given_not_covid =0.04   
pos_not_covid =0.01 -p_covid    


print("Result (calculate_covid_test) ==> : ", calculate_covid_test(p_covid, pos_given_covid, pos_given_not_covid, pos_not_covid))

def defective_product_prob(p_defective, p_pos_give_defective, p_pos_given_not_defective, p_pos_not_defective):
    
    # P(B)=P(B∣A)⋅P(A)+P(B∣¬A)⋅P(¬A)
    pos_b = (p_pos_give_defective * p_defective) + (p_pos_given_not_defective * p_pos_not_defective)


    # P(A|B) =  P(B|A)⋅P(A) / P(B) 

    p_defective_given_pos = (p_pos_give_defective * p_defective)/pos_b 

    return p_defective_given_pos

# 1. probability of defective products
# 2. The probability of defective product being possitive in test
# 3. The probability of not defective product being possitive in test
# 4. probability of not defective products
p_defective = 0.05 
p_pos_give_defective = 0.98 
p_pos_given_not_defective = 0.05 
p_pos_not_defective = 1 - p_defective

print("Result (defective_product_prob) ==> : ", defective_product_prob(p_defective, p_pos_give_defective, p_pos_given_not_defective, p_pos_not_defective))
