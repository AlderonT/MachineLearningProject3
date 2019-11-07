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
<<<<<<< HEAD
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
=======
        let matrixMultiplySIMD (weights : float32[]) (inputValues : float32[]) (outputValues : float32[]) = 

            // If Fused Multiply and Add (FMA) is not on the system, throw error
            if Fma.IsSupported |> not then failwithf "Ed ... ward ... (You don't have the FMA instructions)"

            // Fix the values of the input arguments
            use pweights = fixed weights
            use pinputs = fixed inputValues
            use poutputs = fixed outputValues

            // Create mutable values to be used in the matrix multiplication
            let mutable iw = 0
            let mutable ii = 0
            let mutable io = 0

            // Assuming that outputValues is a multiple of 4, loop
            while io < outputValues.Length do 

                // reset ii to beginning of the input array for the next set of outputs
                ii <- 0 

                // clear acc to Zero
                let mutable acc = Vector128.Zero 

                 // Assuming the inputValues is a multiple of 4, loop for its length
                while ii < inputValues.Length do

                    // fetch a,b,c,d from input array
                    let inputVec = Ssse3.LoadVector128(NativeInterop.NativePtr.add pinputs ii)

                    // shuffle's control byte is interpreted as the index of the element to replicate in each position
                    // 0b00000000uy is the binary literal where we tell the instruction to take the first element and replicate it in the 4 positions
                    // the original Sse Shuffle instruction is weird, because it uses the first argument for the first two destination positions and the second arg for the last two
                    // This means you call Shuffle with the same register twice to get the shuffle we want
                    // likewise 0b00011011 would invert the ordering of the elements since we are putting the last element in pos 0, 3rd element into 1, 2nd to 2 and first into 3
                    // DONT FORGET to put the uy on the end of the number to indicate this is a byte
                    // Sse.Shuffle(inputVec,inputVec,0b00000000uy) -> a,a,a,a

                    let aWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pweights iw)     // load weights
                    let aVec = Sse.Shuffle(inputVec,inputVec,0b00000000uy)                          // get a,a,a,a
                    acc <- Fma.MultiplyAdd(aVec,aWeights,acc)                                       // we now apply a against the a weights for 0 thru 3 outputs
                    iw <- iw+4

                    let bWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pweights iw)
                    let bVec = Sse.Shuffle(inputVec,inputVec,0b01010101uy)                          // get b,b,b,b
                    acc <- Fma.MultiplyAdd(bVec,bWeights,acc)                                       // we now apply b against the b weights for 0 thru 3 outputs
                    iw <- iw+4

                    let cWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pweights iw)
                    let cVec = Sse.Shuffle(inputVec,inputVec,0b10101010uy)                          // get c,c,c,c
                    acc <- Fma.MultiplyAdd(cVec,cWeights,acc)                                       // we now apply c against the c weights for 0 thru 3 outputs
                    iw <- iw+4

                    let dWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pweights iw)
                    let dVec = Sse.Shuffle(inputVec,inputVec,0b11111111uy)                          // get d,d,d,d
                    acc <- Fma.MultiplyAdd(dVec,dWeights,acc)                                       // we now apply d against the d weights for 0 thru 3 outputs
                    iw <- iw+4

                    ii <- ii+4                                                                      // move to next chunk of input's e,f,g,h etc.

                // we know at this point we have processed all the weights for the current output so we can store
                Ssse3.Store(NativeInterop.NativePtr.add poutputs io,acc)

                // move to the next 4 outputs nodes
                io <- io+4 
>>>>>>> acefd6f8a7c1536b9bb712c38f6fd50b7add4fe7




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
