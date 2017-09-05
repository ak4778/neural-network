Neural-network
===

Neural network implements:

* A very simple to use API.
* Automatic data normalization.
* Online training of neural networks in different threads.
* Ability to use the neural network while the system is training it (we train a copy and only later merge the weights).
* Fully connected neural networks using the RPROP (Resilient back propagation) learning algorithm.
* Automatic training with simple overtraining detection.

An example of number array Recognition with blurred array number 7:

cd test

make

./nd


After training:


    1 1 1 1     
    1     1     
    1     1     
    1     1     
    1     1     
    1     1     
    1     1     
    1 1 1 1     

predict number is 0

                
        1       
      1 1       
        1       
        1       
        1       
        1       
                

predict number is 1

                
    1 1 1 1     
          1     
        1       
      1         
    1           
    1 1 1 1     
                

predict number is 2

    1 1 1 1     
          1     
          1     
          1     
    1 1 1 1     
          1     
          1     
    1 1 1 1     

predict number is 3

                
    1           
    1   1       
    1   1       
    1 1 1 1     
        1       
        1       
                

predict number is 4

                
    1 1 1 1     
    1           
    1           
    1 1 1 1     
          1     
          1     
    1 1 1 1     

predict number is 5

                
    1 1 1 1     
    1           
    1           
    1 1 1 1     
    1     1     
    1     1     
    1 1 1 1     

predict number is 6

                
    1 1 1 1 1   
            1   
          1     
        1       
      1         
    1           
                

predict number is 7

                
    1 1 1 1     
    1     1     
    1     1     
    1 1 1 1     
    1     1     
    1     1     
    1 1 1 1     

predict number is 8

                
    1 1 1 1     
    1     1     
    1     1     
    1 1 1 1     
          1     
          1     
    1 1 1 1     

predict number is 9

blurred number 7
                
    1 1 1 1     
            1   
          1     
        1       
      1         
    1           
  1             

predict number is = 7
