//--------------------------------------------------------------------------------------------------------------
//
//  CSCI 447 - Machine Learning
//  Assignment #3, Fall 2019
//  Chris Major
//
//  Functions for matrix multiplication with and without SIMD
//  To be used in implementing the Backpropogation Neural Net
//
//--------------------------------------------------------------------------------------------------------------

// MODULES
//--------------------------------------------------------------------------------------------------------------
namespace Project3

    // Open SIMD Intrinsics modules
    open System
    
    open System.Runtime.Intrinsics
    open System.Runtime.Intrinsics.X86

    // Declare as a module
    module MatrixMultiply =


// FUNCTIONS
//--------------------------------------------------------------------------------------------------------------

        // Function to multiply two matrices matA and matB and store the result in matF
        // Assume that:
        //      - vecA is a size N input vector
        //      - vecF is a size M output vector
        //      - matB is a N x M matrix
        let matrixMultiply (vecA : float32[]) (matB : float32[]) (vecF : float32[]) = 

            // Perform the multiplication for each value of vecF (m)
            vecF
            |> Array.mapi (fun m _  -> 
                
                // Create mutable value for summation
                let mutable sum = 0.f

                // Iterate through each value of vecA (n)
                for n = 0 to vecA.Length - 1 do

                    // Grab the index of the matB that is being multiplied (matIndex)
                    let matIndex = ( ( (vecA.Length) * m) + n)

                    // Multiply and add to previous value
                    sum <- sum + (vecA.[n] * matB.[matIndex])

                // Assign sum to vecF at x
                sum

            )


        // Function to multiply two matrices matA and matB with SIMD commands and store the result in matF
        // Assume that:
        //      - vecA is a size N input vector
        //      - vecF is a size M output vector
        //      - matB is a N x M matrix
        // [NOTE] This is not fully working yet. For testing, use matrixMultiply!
        let matrixmultiplysimd (veca : float32[]) (matb : float32[]) (vecf : float32[]) = 

            // if fused multiply and add (fma) is not on the system, throw error
            if fma.issupported |> not then failwithf "ed ... ward ... (you don't have the fma instructions)"

            // Perform the multiplication for each value of vecF (m)
            vecF
            |> Array.mapi (fun m _  -> 
                
                // Create mutable value for summation
                let mutable sum = 0.f

                // Iterate through each value of vecA (n)
                for n = 0 to vecA.Length - 1 do

                    // 

                    // Multiply and add to previous value
                    let aWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pweights iw)     // Load weights
                    let aVec = Sse.Shuffle(inputVec,inputVec,0b00000000uy)                          // Get a,a,a,a
                    acc <- Fma.MultiplyAdd(aVec,aWeights,acc)                                       // We now apply a against the a weights for 0 thru 3 outputs
                    iw <- iw+4

                // Assign sum to vecF at x
                sum

            )




// IMPLEMENTATIONS AND TESTS
//--------------------------------------------------------------------------------------------------------------

        // Test
        let testVecA = [|2.f; 1.f; 3.f|]
        let testMatB = [|1.f; 2.f; 3.f; 4.f; 5.f; 6.f; 7.f; 8.f; 9.f|]
        let testVecF = Array.init 3 (fun i-> i|>float32)

        matrixMultiply testVecA testMatB testVecF

        printfn "%A" testVecA
        printfn "%A" testMatB
        printfn "%A" testVecF





//--------------------------------------------------------------------------------------------------------------
// END OF CODE
