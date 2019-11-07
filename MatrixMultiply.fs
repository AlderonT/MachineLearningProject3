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
        let matrixMultiplySIMD (veca : float32[]) (matb : float32[]) (vecf : float32[]) = 

            // If Fused Multiply and Add (FMA) is not on the system, throw error
            if Fma.IsSupported |> not then failwithf "Ed ... ward ... (You don't have the FMA instructions)"

            // Fix the values of the input arguments so the array addresses are pinned.
            use pB = fixed vecB
            use pA = fixed vecA
            use pF = fixed vecF

            // Create mutable values to be used in the matrix multiplication
            let mutable iB = 0
            let mutable iA = 0
            let mutable iF = 0

            // Assume that outputValues is a multiple of 4
            while iF < vecF.Length do 

                // Reset iA (vecA counter) to beginning of the input array (vecA) for the next set of outputs (vecF)
                iA <- 0 

                // Clear accumulator to zero
                let mutable acc = Vector128.Zero 

                // Iterate through the input vector
                // Assume that vecA is a multiple of 4
                while iA < vecA.Length do 

                    // Fetch a, b, c, and d from vecA
                    let inputVec = Ssse3.LoadVector128(NativeInterop.NativePtr.add pA iA)

                    // Perform intrinsics on a
                    let aWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pB iB)               // Load weights
                    let aVec = Sse.Shuffle(inputVec, inputVec, 0b00000000uy)                            // Get a, a, a, a
                    acc <- Fma.MultiplyAdd(aVec, aWeights, acc)                                         // Apply a against the a weights for 0 to 3 outputs
                    iB <- iB + 4                                                                        // Increment vecB counter by 4

                    // Perform intrinsics on b
                    let bWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pB iB)               // Load weights
                    let bVec = Sse.Shuffle(inputVec, inputVec, 0b01010101uy)                            // Get b, b, b, b
                    acc <- Fma.MultiplyAdd(bVec, bWeights, acc)                                         // Apply b against the b weights for 0 to 3 outputs
                    iB <- iB + 4                                                                        // Increment vecB counter by 4

                    // Perform intrinsics on c
                    let cWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pB iB)               // Load weights
                    let cVec = Sse.Shuffle(inputVec,inputVec,0b10101010uy)                              // Get c, c, c, c
                    acc <- Fma.MultiplyAdd(cVec, cWeights, acc)                                         // Apply c against the c weights for 0 to 3 outputs    
                    iB <- iB + 4                                                                        // Increment vecB counter by 4

                    // Perform intrinsics on d
                    let dWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pB iB)               // Load weights
                    let dVec = Sse.Shuffle(inputVec, inputVec,0b11111111uy)                             // Get d, d, d, d
                    acc <- Fma.MultiplyAdd(dVec, dWeights, acc)                                         // Apply d against the d weights for 0 to 3 outputs            
                    iB <- iB + 4                                                                        // Increment vecB counter by 4

                // Store all values for the current output of vecF
                Ssse3.Store(NativeInterop.NativePtr.add pF iF, acc)

                // Increment output vector (vecF) counter by 4
                iF <- iF + 4 


// IMPLEMENTATIONS AND TESTS
//--------------------------------------------------------------------------------------------------------------

        // Create test vectors
        let testVecA = [|2.f; 1.f; 3.f|]
        let testMatB = [|1.f; 2.f; 3.f; 4.f; 5.f; 6.f; 7.f; 8.f; 9.f|]
        let testVecFRegular = Array.init 3 (fun i-> i|>float32)
        let testVecFSIMD = Array.init 3 (fun i-> i|>float32)

        // Perform test multiplications
        matrixMultiply testVecA testMatB testVecFRegular
        matrixMultiplySIMD testVecA testMatB testVecFSIMD

        // Print results
        printfn "%A" testVecA
        printfn "%A" testMatB
        printfn "%A" testVecF


//--------------------------------------------------------------------------------------------------------------
// END OF CODE
