// Learn more about F# at http://fsharp.org

open System

open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86

module X =
    let time cnt f =
        let sw = System.Diagnostics.Stopwatch.StartNew()
        for i = 1 to cnt do
            f()
        sw.Stop()
        (sw.Elapsed.TotalSeconds*1000000.)/float cnt



    if Fma.IsSupported |> not then failwithf "You don't have the FMA instructions"

    // the key with SIMD is to figure out a way to organize the data so that you can perform the series of computations in parallel
    // if you have access to the FMA (Fused Multiply and Add) instructions then you can perform a multiply and addition in one step
    // This requires you to process node values and weights in a different pattern than you might normally do
    // you want to treat each section of the register as a specific output node computation and then move through the weights that way
    // Example:
    //  you have 8 input nodes going to 8 output nodes, this means you have 64 weights. Input nodes are labeled a,b,c,d,e,f,g,h and
    //  the output nodes 0,1,2,3,4,5,6,7. Weights will be labeled a0 which is the weight connecting a to 0
    //
    // To reduce load and stores to the minium you might do the following:
    // You have two float32 arrays one for the input values and one for the output values
    // Make sure your input, output and temp arrays are multiples of the element size 4(128bit) or 8(256bit). Remember 0 weights are
    // equivelent to not being connected so you can removed the excess nodes by making their weights zero and then preventing
    // backpropagation from affecting weights that are to phantom nodes. We do this to make the algorithm simpler.
    // We will assume a Vector128, this means we can work on up to output 4 nodes at a time in this example we will work on 0,1,2,3
    // Since we want to reduce the memory accesses to a mimimum and take advantage of cache we will process the 0,1,2,3 for each input
    // node, we will fetch those also 4 at a time into a Vector a,b,c,d. We will use the Shuffle Instructions to move them into a
    // register with the same component replicated across elements to use with the FMA against each set of weights
    // the accumulator register will hold the results until we are done with the correct number of input nodes and then we store that
    // vector into the output array as a completed weight calculation.
    //
    // This should mean 
    // 
    // the input array looks like this [|a;b;d;c;e;f;g;h|]
    // the output array looks like [|0;0;0;0;0;0;0;0|]
    // we are using this organization to take advantage of the FMA instruction so we might want use Vector256 anyway
    // the weights should be layed out like this (with zero weight for not existing nodes on either side) for Vector128, this changes if you can use 256 which is twice as fast for this purpose:
    // let weights = [|
    //      (*0123*) a0;a1;a2;a3;b0;b1;b2;b3;c0;c1;c2;c3;d0;d1;d2;d3;e0;e1;e2;e3;f0;f1;f2;f3;g0;g1;g2;g3;h0;h1;h2;h3 (*at this point 0 thru 3 are complete so we can store them*)
    //      (*4567*) a4;a5;a6;a7;b4;b5;b6;b7;c4;c5;c6;c7;d4;d5;d6;d7;e4;e5;e6;e7;f4;f5;f6;f7;g4;g5;g6;g7;h4;h5;h6;h7 (*at this point 4 thru 7 are complete so we can store them*)
    // |]


    let inputValues = Array.init 8 (fun i -> if i < 5 then float32 (i+1) else 0.f)
    let outputValues : float32[] = Array.zeroCreate 4
    let weights =
        [
            1.00f,0.25f,0.25f,0.25f  // this is the values for the a column in the example excel document
            0.25f,1.00f,0.25f,0.25f  // this is the values for the b column in the example excel document
            0.25f,0.25f,0.25f,0.25f  // etc..
            0.25f,0.25f,0.25f,0.25f
            0.25f,0.25f,0.25f,0.25f  // this is the values for the e column in the second example in excel
            0.00f,0.00f,0.00f,0.00f  // we don't have a f,g,h input node but we set the input values and
            0.00f,0.00f,0.00f,0.00f  // weights to zero
            0.00f,0.00f,0.00f,0.00f
        ]
        |> Seq.collect (fun (a0,a1,a2,a3) -> [a0;a1;a2;a3])
        |> Seq.toArray

    // This code takes the following number of instructions to complete:
    // n = number of input nodes
    // m = number of output nodes
    // n*m = number of weights
    // 9nm/8 + m/2 instructions so roughly nm+m/2 so n*m so the time complexity is O(nm)
    // if we used 256bits
    // we would get:
    // 17nm/32 + m/4 so roughly nm/2 + m/4 so (n*m)/2 so again time complexity is O(nm) but this uses half the number of instructions to do the same work.
    // given clock rate at full load using SIMD would most likely be 2.5 GHz the FMA takes 5 clocks on a Skylake but if properly queued can be completed in 1/2 a clock
    // so you can process a 1000x1000 layer interconnect in around 
    

    let computeOutput (weights:float32[]) (inputValues:float32[]) (outputValues:float32[]) =
        use pweights = fixed weights
        use pinputs = fixed inputValues
        use poutputs = fixed outputValues
        let mutable iw = 0
        let mutable ii = 0
        let mutable io = 0
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


    // the following example ran in 1.86 us that is running 100 input nodes through a 10000 element weights array
    if (false) then
        let rnd = System.Random()
        let inputValues2 = Array.init 100 (fun _ -> rnd.NextDouble() |> float32)
        let outputValues2 : float32[] = Array.zeroCreate 100
        let weights2 = Array.init (inputValues2.Length*outputValues2.Length) (fun _ -> rnd.NextDouble() |> float32)

        time 10000 (fun _ -> computeOutput weights2 inputValues2 outputValues2)
        |> ignore

    // the following example ran in 235.53 us that is running 1000 input nodes through a 1000000 element weights array
    if (false) then
        let rnd = System.Random()
        let inputValues2 = Array.init 1000 (fun _ -> rnd.NextDouble() |> float32)
        let outputValues2 : float32[] = Array.zeroCreate 1000
        let weights2 = Array.init (inputValues2.Length*outputValues2.Length) (fun _ -> rnd.NextDouble() |> float32)

        time 10000 (fun _ -> computeOutput weights2 inputValues2 outputValues2)
        |> ignore

// creating 256bit vectors of integers allows working on 8 integers at a time

let a = Vector256.Create(1,2,3,4,5,6,7,8)
let b = Vector256.Create(9,10,11,12,13,14,15,16)

// documenting how the Horitzontal Add instruction works
//HorizontalAdd (a,b,c,d) (e,f,g,h) -> (a+b,c+d,e+f,g+h)
//HorizontalAdd (a,b,c,d,e,f,g,h) (i,j,k,l,m,n,o,p) -> (a+b,c+d,i+j,k+l,e+g,g+h,m+n,o+p)


// The following is sample implementation of using the SIMD instructions to perform loop unrolling for summing an integer array using both 128bit and 256bit instructions.
// Your computer might not be able to execute the sum256 unless it is relatively new circa 2016+

let sum128 (source:int[]) =
    let mutable vresult = Vector128<int>.Zero
    let lastBlockIndex = source.Length - (source.Length%4);
    use pSource = fixed source
    let mutable i = 0
    while i < lastBlockIndex do        
        vresult <- Sse2.Add(vresult,Sse2.LoadVector128(NativeInterop.NativePtr.add pSource i)) // here we compute the new pointer address pSource + i this get removed in the compilation,
                                                                                               // and then we Load a Vecto128 of int32 into a register from memory. Finally we actually
                                                                                               // perform the PADDD operation that adds the two and we put it back into vresult temp register
        i <- i + 4
    vresult <- Ssse3.HorizontalAdd(vresult,vresult) // combine a,b,c,d into (a+b,c+d,a+b,c+d)  this is the Cool PHADDD instruction that add the two arguments pairwise and puts them into the out register
    vresult <- Ssse3.HorizontalAdd(vresult,vresult) // combine (a+b,c+d,a+b,c+d) into (a+b+c+d,a+b+c+d,a+b+c+d,a+b+c+d) we do the operation on itself again to get the full addition
    let mutable result = vresult.ToScalar() // finally we extract the first element of the 128bit vector as an int32
    while i < source.Length do  // complete the sum by going over the remaining elements we might have missed
        result <- result + NativeInterop.NativePtr.get pSource i
        i <- i+1
    result

let sum256 (source:int[]) =
    if Avx.IsSupported |> not then
        failwithf "You dont' have the Avx instructions"
    if Avx2.IsSupported |> not then
        failwithf "You dont' have the Avx instructions"
    let mutable vresult = Vector256<int>.Zero
    let lastBlockIndex = source.Length - (source.Length%8);
    use pSource = fixed source
    let mutable i = 0
    while i < lastBlockIndex do        
        vresult <- Avx2.Add(vresult,Avx.LoadVector256(NativeInterop.NativePtr.add pSource i))
        i <- i + 8
                                                   //         a,b,c,d,e,f,g,h a,b,c,d,e,f,g,h -> ab cd ab cd ef gh ef gh
    vresult <- Avx2.HorizontalAdd(vresult,vresult) // combine a,b,c,d,e,f,g,h into (a+b,c+d,a+b,c+d,e+f,g+h,e+f,g+h)    the Avx2's VPHADDD instruction basically does two of the Ssse3 together so you get a slightly different pattern
                                                   //         ab cd ab cd ef gh ef gh ab cd ab cd ef gh ef gh -> abcd abcd abcd abcd efgh efgh efgh efgh
    vresult <- Avx2.HorizontalAdd(vresult,vresult) // combine (a+b,c+d,a+b,c+d,e+f,g+h,e+f,g+h) into (a+b+c+d,a+b+c+d,a+b+c+d,a+b+c+d,e+f+g+h,e+f+g+h,e+f+g+h,e+f+g+h)
    // at this point another Horitonzal add won't work since we'd only get the first and second halfs double again. So we instead extract the low and high 128bits and then perform a normal Sse2 PADDD to combine the sums
    let v128Result = Sse2.Add(Avx2.ExtractVector128(vresult,0uy),Avx2.ExtractVector128(vresult,1uy)) // (a+b+c+d+e+f+g+h,a+b+c+d+e+f+g+h,a+b+c+d+e+f+g+h,a+b+c+d+e+f+g+h)
    let mutable result = v128Result.ToScalar() // extract the first element as the partial sum
    while i < source.Length do  // complete the sum by going over the remaining elements we might have missed
        result <- result + NativeInterop.NativePtr.get pSource i
        i <- i+1
    result


[<EntryPoint>]

let main argv =
    if Fma.IsSupported |> not then
        failwithf "You don't have the FMA instructions"
    if Ssse3.IsSupported |> not then
        failwithf "You don't have the Ssse3 instructions"
    let data = Array.init 200 id
    let x = sum128 data
    X.computeOutput X.weights X.inputValues X.outputValues
    printfn "inputs: %A" X.inputValues
    printfn "outputs: %A" X.outputValues
    printfn "%d" x
    0 // return an integer exit code
