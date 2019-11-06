//--------------------------------------------------------------------------------------------------------------
//
//  CSCI 447 - Machine Learning
//  Assignment #3, Fall 2019
//  Chris Major
//
//  Basic function to compute the activation functions of the neural network
//  To be used in implementing the Backpropogation Neural Net
//
//--------------------------------------------------------------------------------------------------------------

// MODULES
//--------------------------------------------------------------------------------------------------------------
namespace Project3

    // Declare as a module
    module ActivationFunction = 


// FUNCTIONS
//--------------------------------------------------------------------------------------------------------------

        // Function for the sigmoidal logistic function
        let sigmoidLogisticFunction x = 
            (1.0 / (1.0 + System.Math.E ** ( (float) -x)))

        // Function for the sigmoidal hyperbolic tangent
        let sigmoidHyperbolicTangent x = 
            System.Math.Tanh ((float) x)
    
        // Function for the inverse square root unit (ISRU)
        let ISRU x = 
            1


         // Make sigmoids SIMD [CHRIS]      

// IMPLEMENTATIONS
//--------------------------------------------------------------------------------------------------------------

    // none

//--------------------------------------------------------------------------------------------------------------
// END OF CODE