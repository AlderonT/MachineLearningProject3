//--------------------------------------------------------------------------------------------------------------
//
//  CSCI 447 - Machine Learning
//  Assignment #3, Fall 2019
//  Chris Major
//
//  Basic function to compute the dot product of two vectors
//  To be used in implementing the Backpropogation Neural Net
//
//--------------------------------------------------------------------------------------------------------------

// MODULES
//--------------------------------------------------------------------------------------------------------------
namespace Project3

    // Declare as a module
    module DotProduct =


// FUNCTIONS
//--------------------------------------------------------------------------------------------------------------

        // Function to take the dot product of two vectors vecA and vecB
        let dotProduct vecA vecB =
            Seq.zip vecA vecB                           // Zip the values of each sequence into tuples
            |> Seq.map (fun (a, b) -> (a * b))          // Iterate through each tuple and multiply the values
            |> Seq.sum                                  // Add the multiplied values together and return


// IMPLEMENTATIONS AND TESTS
//--------------------------------------------------------------------------------------------------------------

        // Create test vectors as sequences
        //let testA = seq {1; 1; 1; 1}
        //let testB = seq {2; 3; 4; 5}

        //// Perform the dot multiplication on the test sequences
        //let testDotProduct = dotProduct testA testB

        //// Print to report results
        //printf "RESULTS:\n"
        //printf "A = %A\n" testA
        //printf "B = %A\n" testB
        //printf "A dot B = %A\n" testDotProduct

//--------------------------------------------------------------------------------------------------------------
// END OF CODE
