
import numpy as np

training_data = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics' +
                        '/Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
                        '/mnist_train_images.npy');
training_labels = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics' +
                        '/Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
                        '/mnist_train_labels.npy');

validation_data = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics' +
                        '/Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
                        '/mnist_validation_images.npy');
validation_labels = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics' +
                        '/Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
                        '/mnist_validation_labels.npy');

test_data = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics' +
                        '/Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
                        '/mnist_test_images.npy');
test_labels = np.load('/home/archit/Archit_Kumar/Documents/WPI/Academics' +
                        '/Spring 2019 Subjects/Deep_learning/Hw3/MNIST_files' +
                        '/mnist_test_labels.npy');

#Considering only training data first
X = training_data;  # Size 55000 by 784(m by n)
y = training_labels;    # Size 55000 by 10(m by c)
# Note: y or class labels are in binary form already!

m = X.shape[0]

X = np.hstack([np.ones((X.shape[0],1)), X]);    # Size 55000 by 785

#hidden_layers = input('Enter the number of hidden layers required: ');
#units_hiddenlayer = input('Enter the number of units reequired per hidden ' +
                            #'layer: ');
hidden_layers = 1;
units_hiddenlayer = 25;

# Regularization strength parameter
Lambda = 1; 

# Randomly initialising Theta values
const = 0.12;   # Decides range of random initialised values for weights 
Theta1 = np.random.rand(units_hiddenlayer, X.shape[1]) * 2 * const - const; # size 25 by 785
Theta2 = np.random.rand(y.shape[1], units_hiddenlayer + 1) * 2 * const - const; # size 10 by 26

# Initializing accumulator values
D_1 = np.zeros(Theta1.shape);   # size 25 by 785
D_2 = np.zeros(Theta2.shape);   # size 10 by 26

# Performing forward and backward prop one example at a time
for t in range(0,X.shape[0]):
    
    # Forward propagation
    
    # Already appended bias to input layer
    a_1 = X[t, :];  # size 1 by 785
    z_2 = np.dot( a_1, Theta1.T )    # size 1 by 25
    # Using the RELU function
    a_2 = np.maximum(z_2, 0);   # size 1 by 25
    
    #Appending the bias to layer2
    a_2 = np.hstack([1, a_2]);  # size 1 by 26
    z_3 = np.dot( a_2, Theta2.T );  # size 1 by 10 
    # Calculating softmax to get prediction
    a_3 = np.exp(z_3) / ( np.sum(np.exp(z_3)) );
    
    # Output of 3 layer neural network
    hypothesis = a_3 ;
    
    # Converting activations to column vector form from row vector form
   
    a_1 = a_1.T;   # size 785 by 1
    a_1 = a_1.reshape(a_1.shape[0],1);
    a_2 = a_2.T;   # size 26 by 1
    a_2 = a_2.reshape(a_2.shape[0],1);
    a_3 = a_3.T;   # size 10 by 1
    a_3 = a_3.reshape(a_3.shape[0],1);
    hypothesis = hypothesis.T; # size 10 by 1
    hypothesis = hypothesis.reshape(hypothesis.shape[0],1);
    
    # Backward propagation
    
    temp_y = (y[1,:]).T;
    temp_y = temp_y.reshape(temp_y.shape[0],1);
    d_3 = hypothesis - temp_y;   # size 10 by 1
    gprime_2 = a_2 * (1 - a_2); # size 26 by 1
    d_2 = np.dot( Theta2.T, d_3 ) * gprime_2;    # size 26 by 1
    d_2 = d_2.reshape(d_2.shape[0],1);
    d_2 = d_2[1:];  # Because we do not want to include bias in delta values! 
    # Note: size of d_2 changed to 25 by 1
    
    # Now to update the accumulator
    
    D_1 = D_1 + np.dot( d_2, a_1.T );  # size 25 by 785
    D_2 = D_2 + np.dot( d_3, a_2.T );  # size 10 by 26
    
#Moving out of loop

temp1 = Theta1; temp2 = Theta2;
temp1[:, 0] = 0; temp2[:, 0] = 0;

Theta1_grad = (1/m) * ( D_1 + Lambda * temp1);
Theta2_grad = (1/m) * ( D_2 + Lambda * temp2);

# Perfomring flattening operation
Theta1_grad = np.matrix.flatten(Theta1_grad);
Theta1_grad = Theta1_grad.reshape(Theta1_grad.shape[0], 1);
Theta2_grad = np.matrix.flatten(Theta2_grad);
Theta2_grad = Theta2_grad.reshape(Theta2_grad.shape[0], 1);

gradient = np.vstack([Theta1_grad, Theta2_grad]);
    
#print(gradient[1:10]);

    
     
    
    
    
     
    
    



    
