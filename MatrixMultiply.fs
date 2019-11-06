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

    // Open SIMD Intrinsics modules
    open System
    
    open System.Runtime.Intrinsics
    open System.Runtime.Intrinsics.X86

    // Declare as a module
    module MatrixMultiply =


// FUNCTIONS
//--------------------------------------------------------------------------------------------------------------

        // Function to multiply two matrices matA and matB and store the result in matF
        let matrixMultiply (matA : float32[]) (matB : float32[]) (matF : float32[]) = 

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


        // Function to multiply two matrices matA and matB with SIMD commands and store the result in matF
        let matrixMultiplySIMD (matA : float32[]) (matB : float32[]) (matF : float32[]) = 

            // If Fused Multiply and Add (FMA) is not on the system, throw error
            if Fma.IsSupported |> not then failwithf "Ed ... ward ... (You don't have the FMA instructions)"

            // Fix the values of the input arguments
            use pmatA = fixed matA
            use pmatB = fixed matB
            use pmatF = fixed matF

            // Create mutable values to be used in the matrix multiplication
            let mutable iA = 0
            let mutable iB = 0
            let mutable iF = 0

            while io < outputValues.Length do // we assume that outputValues is a multiple of 4
                ii <- 0 // reset ii to beginning of the input array for the next set of outputs
                let mutable acc = Vector128.Zero // clear acc to Zero
                while ii < inputValues.Length do // we assume that inputValues is a multiple of 4
                    // fetch a,b,c,d from input array
                    let inputVec = Ssse3.LoadVector128(NativeInterop.NativePtr.add pinputs ii)
                    // shuffle's control byte is interpreted as the index of the element to replicate in each position
                    // 0b00000000uy is the binary literal where we tell the instruction to take the first element and replicate it in the 4 positions
                    // the original Sse Shuffle instruction is weird, because it uses the first argument for the first two destination positions and the second arg for the last two
                    // This means you call Shuffle with the same register twice to get the shuffle we want
                    // likewise 0b00011011 would invert the ordering of the elements since we are putting the last element in pos 0, 3rd element into 1, 2nd to 2 and first into 3
                    // DONT FORGET to put the uy on the end of the number to indicate this is a byte
                    // Sse.Shuffle(inputVec,inputVec,0b00000000uy) -> a,a,a,a
                    let aWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pweights iw) // load weights
                    let aVec = Sse.Shuffle(inputVec,inputVec,0b00000000uy) // get a,a,a,a
                    acc <- Fma.MultiplyAdd(aVec,aWeights,acc) // we now apply a against the a weights for 0 thru 3 outputs
                    iw <- iw+4
                    let bWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pweights iw)
                    let bVec = Sse.Shuffle(inputVec,inputVec,0b01010101uy) // get b,b,b,b
                    acc <- Fma.MultiplyAdd(bVec,bWeights,acc) // we now apply b against the b weights for 0 thru 3 outputs
                    iw <- iw+4
                    let cWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pweights iw)
                    let cVec = Sse.Shuffle(inputVec,inputVec,0b10101010uy) // get c,c,c,c
                    acc <- Fma.MultiplyAdd(cVec,cWeights,acc) // we now apply c against the c weights for 0 thru 3 outputs
                    iw <- iw+4
                    let dWeights = Ssse3.LoadVector128(NativeInterop.NativePtr.add pweights iw)
                    let dVec = Sse.Shuffle(inputVec,inputVec,0b11111111uy) // get d,d,d,d
                    acc <- Fma.MultiplyAdd(dVec,dWeights,acc) // we now apply d against the d weights for 0 thru 3 outputs
                    iw <- iw+4
                    ii <- ii+4 // move to next chunk of input's e,f,g,h etc.
                // we know at this point we have processed all the weights for the current output so we can store
                Ssse3.Store(NativeInterop.NativePtr.add poutputs io,acc)
                io <- io+4 // move to the next 4 outputs nodes
            // we're done, the output array should have all the calculated weights




// IMPLEMENTATIONS AND TESTS
//--------------------------------------------------------------------------------------------------------------

        // Test
        let x = sum128 data
        X.computeOutput X.weights X.inputValues X.outputValues
        printfn "inputs: %A" X.inputValues
        printfn "outputs: %A" X.outputValues
        printfn "%d" x


//--------------------------------------------------------------------------------------------------------------
// END OF CODE
