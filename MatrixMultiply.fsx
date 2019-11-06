//--------------------------------------------------------------------------------------------------------------
//
//  CSCI 447 - Machine Learning
//  Assignment #3, Fall 2019
//  Chris Major
//
//  Functions for matrix multiplication using SIMD
//  To be used in implementing the Backpropogation Neural Net
//
//--------------------------------------------------------------------------------------------------------------

// MODULES
//--------------------------------------------------------------------------------------------------------------
namespace Project3

    // Declare as a module
    module MatrixMultiply =


// FUNCTIONS
//--------------------------------------------------------------------------------------------------------------

        // Function to take the dot product of two vectors vecA and vecB
        let matrixMultiply (matA : float32[]) (matB : float32[]) = 

            // Convert each float array into a "matrix"
            // Column first, not row first
            let realMatrixA =
                matA
                |> Seq.collect (fun (a0,a1,a2,a3) -> [a0;a1;a2;a3])
                |> Seq.toArray

            let realMatrixB =
                matB
                |> Seq.collect (fun (a0,a1,a2,a3) -> [a0;a1;a2;a3])
                |> Seq.toArray


            // The slow version



            // Reformat matrices, refer to Program.fsx
            // Pull apart and multiply
            // There is no spoon

            1


// IMPLEMENTATIONS AND TESTS
//--------------------------------------------------------------------------------------------------------------

        // Test

//--------------------------------------------------------------------------------------------------------------
// END OF CODE
