def test_rates_positive(rates): 
    #test if rates are all positive
    for i in range(len(rates)):
        for j in range(len(rates)):
            if rates[i][j] < 0:
                return False

def test_rates_square(rates): 
    #test if rates are all positive
    if len(rates) != len(rates[0]):
        return False