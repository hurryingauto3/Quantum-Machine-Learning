def GradDesc(start, rate, precision, previous_step_size, max_iters):
    cur_x = start # The algorithm starts at x=3
    rate = rate # Learning rate
    precision = precision #This tells us when to stop the algorithm
    previous_step_size = previous_step_size #
    max_iters = max_iters # maximum number of iterations
    iters = 0 #iteration counter
    df = lambda x: 2*(x+5) #Gradient of our function 


    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x #Store current x value in prev_x
        cur_x = cur_x - rate * df(prev_x) #Grad descent
        previous_step_size = abs(cur_x - prev_x) #Change in x
        iters = iters+1 #iteration count
        print("Iteration",iters,"\nX value is",cur_x) #Print iterations
        
    print("The local minimum occurs at", cur_x)


GradDesc(3, 0.01, 0.01, 1, 100000)