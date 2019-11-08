//--------------------------------------------------------------------------------------------------------------
//
//  CSCI 447 - Machine Learning
//  Assignment #3, Fall 2019
//  Chris Major
//
//  Basic function to compute the Gaussian
//  To be used in implementing the RBF Net
//
//--------------------------------------------------------------------------------------------------------------

// MODULES
//--------------------------------------------------------------------------------------------------------------
namespace Gaussian

    // Declare as a module
    module GaussianFunction = 


// FUNCTIONS
//--------------------------------------------------------------------------------------------------------------

        // Function for the sigmoidal logistic function
        let gaussianFunction (x : float32[]) (xPrime : float32[]) (sigma : float32) = 
            
            // Calculate the distance of the two points
            let distanceX = 
                Seq.zip x xPrime
                |> Seq.sumBy (fun (a,b) -> (a - b) * (a - b) )

            // Return the exponential of the square Euclidean distance over the sigma term
            System.Math.Exp(float (-1.0f * (distanceX * distanceX) * (2.0f * sigma * sigma))) |> float32


// IMPLEMENTATIONS
//--------------------------------------------------------------------------------------------------------------

    // none

//--------------------------------------------------------------------------------------------------------------
// END OF CODE