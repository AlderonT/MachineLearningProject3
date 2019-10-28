//--------------------------------------------------------------------------------------------------------------
//
//  CSCI 447 - Machine Learning
//  Assignment #3, Fall 2019
//  Chris Major, Farshina Nazrul-Shimim, Tysen Radovich, Allen Simpson
//
//  [DESCRIPTION]
//
//--------------------------------------------------------------------------------------------------------------

// MODULES
//--------------------------------------------------------------------------------------------------------------
namespace Project3

    // Load the file names from the same folder
    #load "ActivationFunction.fsx"
    #load "DotProduct.fsx"

    // Open the modules
    open ActivationFunction
    open DotProduct

    // Declare as a module
    module Assignment3 = 


// OBJECTS
//--------------------------------------------------------------------------------------------------------------

        // Create a Layer object to represent a layer within the neural network
        type Layer = 
            abstract member nodes           : float32[]                         // Sequence to make up vectors
            abstract member nodeCount       : int                               // Number of nodes in the layer

        // Create a ConnectionMatrix object to represent the connection matrix within the neural network
        type ConnectionMatrix = 
            abstract member weights         : float32[]                         // Sequence of weights within the matrix
            abstract member inputLayer      : Layer                             // Input layer
            abstract member outputLayer     : Layer                             // Output layer


// FUNCTIONS
//--------------------------------------------------------------------------------------------------------------

        // Function to run the Feedforward Network with Backpropogation
        let FFNN inputs hiddenLayers hiddenNodes outputs = 
            1                                                               // Write out functions


        // Function to run the Radial Basis Function Network
        let RBF inputs gaussianBasisFn outputs addFFNN = 
            1                                                               // Write out functions


// IMPLEMENTATIONS
//--------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------
// END OF CODE