//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
//
//  CSCI 447 - Machine Learning
//  Assignment #4, Fall 2019
//  Chris Major, Farshina Nazrul-Shimim, Tysen Radovich, Allen Simpson
//
//  Functions for the implementation of several population algorithms to train a feedforward neural network
//
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 


// NAMESPACE
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
namespace SAE_NN
(*
#load "C:/work/snippets/Clipboard.fsx"
open Clipboard
*)

// Open modules from local directory
open Types
open Datasets


// AUTOENCODER MODULE
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
module Autoencoder =

    // Function to generate random values over a uniform distribution
    let rand_uniform_1m_1 =
        let rand = System.Random()                                                          // Create RNG
        fun () ->
            let n = rand.NextDouble()                                                       // Generate the next random double value
            float32 (n*2. - 1.)                                                             // Return the normalized value as a float



    // TYPE DECLARATIONS
    //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    
    // Type to implement a biased layer, which adds an extra bias value of 1.f as the last node in the layer
    type Layer =
        {
            nodes: float32[]                                                                // Array of layer nodes    
            deltas: float32[]                                                               // Array of delta errors
        }
        with
            member this.Length = this.nodes.Length                                          // Method to the length of a layer
            member this.ResetBias() = this.nodes.[this.nodes.Length-1] <- 1.f               // Method to reset the bias nodes
            static member Create size =                                                     // Method to create a Layer with an extra bias node
                let x = {
                    nodes = Array.zeroCreate (size+1)                                       // Increase size of nodes array
                    deltas = Array.zeroCreate (size+1)                                      // Increase size of deltas array
                }
                x.nodes.[x.nodes.Length-1] <- 1.f                                           // Set bias nodes
                x                                                                           // Return new array
    
    // Type to implement a connection matrix
    type ConnectionMatrix =
        {
            inputLayer: Layer                                                               // Input layer
            outputLayer: Layer                                                              // Output layer 
            weights: float32[,]                                                             // Weight matrix
        }

    // Type to implement a biased network
    type Network = 
        {           
            connections: ConnectionMatrix[]                                                 // Connection matrix
            expectedOutputs: float32[]                                                      // Array of expected outputs
        }
        with
            
            // Method to validate proper layer specifications for the network
            member this.featureLayer =
                if this.connections.Length % 2 <> 0 then
                    failwithf "ERROR: To extract features, the network must have an odd number of layers!"
                else
                    let cm = this.connections.[this.connections.Length/2 - 1]
                    cm.outputLayer

            // Method to get the output layer of the network
            member this.outputLayer = this.connections.[this.connections.Length-1].outputLayer

            // Method to get the input layer of the network
            member this.inputLayer = this.connections.[0].inputLayer

            // Methods to load in the data we need
            // #1: Only copy in the non-bias elements, leave the last bias node alone
            member this.LoadInput (inputs:float32[]) =
                System.Array.Copy(inputs,this.inputLayer.nodes,this.inputLayer.Length-1)
                
            // #2: Only copy in the non-bias elements, leave the last bias node alone
            member this.LoadInput (inputs:seq<float32>) =
                inputs |> Seq.iteri (fun i v -> if i < (this.inputLayer.nodes.Length-1) then this.inputLayer.nodes.[i] <- v)

            // #3: When loading expected outputs into the network we remember to set the bias node entry to 1.f
            member this.LoadExpectedOutput (inputs:float32[]) =
                System.Array.Copy(inputs,this.expectedOutputs,this.expectedOutputs.Length-1)
                this.expectedOutputs.[this.expectedOutputs.Length-1] <- 1.f

            // #4: The expected output does have a bias node because it makes it easier to write the code, we just have to remember to ignore it
            member this.LoadExpectedOutput (inputs:seq<float32>) =
                inputs |> Seq.iteri (fun i v -> if i < (this.inputLayer.nodes.Length-1) then this.expectedOutputs.[i] <- v)
                this.expectedOutputs.[this.expectedOutputs.Length-1] <- 1.f

            // Static method to perform a customized Seq.map function
            static member Create (nodeValue:unit->float32) (sizes:int seq) =
                let e = sizes.GetEnumerator()                                               // we are creating an enumerator enumerating over sizes
                if e.MoveNext() |> not then                                     //if have a size<1 then we
                    failwithf "need at least two sizes specified"               //fail
                let inputLayer = Layer.Create e.Current                         //here we initiate a layer using our enumerator's current element
                let rec loop inLayer (connections:ResizeArray<_>) =             //rec do:
                    if e.MoveNext() |> not then connections.ToArray()           //if you can't move on, return the finished Array
                    else                                                        //else
                        let outLayer = Layer.Create e.Current                   //create the output layer
                        let cm = {                                              //create an instance of the connection matrix
                            inputLayer = inLayer
                            outputLayer = outLayer
                            // we subtract 1 from the length of the outLayer length to account for the bias entry that we DONT write to
                            // both inLayer and output layers get an extra bias node that is already included
                            
                            //Our initial weights are determined here: the size of the weights matrix takes into account that hte output layer should not write over the bias node.
                            weights = Array2D.init (outLayer.Length-1) inLayer.Length (fun _ _ -> nodeValue())  
                        }
                        connections.Add(cm)                                     //we insert the connection matrix into connections
                        loop outLayer connections                               //now the next input layer is the current output layer
                let connections = loop inputLayer (ResizeArray())               //get the value from this recursive call and stick it in connectsion
                if connections.Length = 0 then                                  //if our thing is too small size < 2
                    failwithf "need at least two sizes specified"               //fail
                {                                                               //then we return our connections and our expected outputs in a Network object
                    connections = connections
                    expectedOutputs = Array.zeroCreate connections.[connections.Length-1].outputLayer.Length
                }
            member this.Save() =                                                                                        //this is a save fn here we are using some low-level stuff to write to memory as base 64 so we can copy-paste networks for testing
                use ms = new System.IO.MemoryStream()                                                                   //memory stream
                use cs = new System.IO.Compression.DeflateStream(ms,System.IO.Compression.CompressionLevel.Optimal)     //stream compressor
                use bw = new System.IO.BinaryWriter(cs)                                                                 //and a binary writer
                let sizes =                                                 //get the sizes of the input, hidden layers, and outputlayer
                    seq {                                                   //as a seq
                        yield!                                                      
                            this.connections                                // containing the connecctions
                            |> Seq.map (fun cm -> cm.inputLayer.Length-1)   // we subtract one so that we get the original non-biased lengths for the layers
                        yield this.outputLayer.Length-1                     // and containing our output layer without the bias layer
                    }
                    |> Seq.toArray                                          //return as an array
                // write the sizes out first
                bw.Write(sizes.Length)
                sizes
                |> Seq.iter (bw.Write)
                this.connections                                            //here we are writing the connections in 1st to last row-major order to the memory streme (same direction as construction
                |> Seq.iter (fun cm ->
                    // now write out the elements
                    Array2D.iter (fun x -> bw.Write(x:float32)) cm.weights
                )
                bw.Flush()
                bw.Close()
                let bytes = ms.ToArray()
                System.Convert.ToBase64String bytes                         //return the memory stream as a string in base 64

            static member Load(serialized:string) =                                                                     //Load undoes save by going backwards and producing a network
                let bytes = System.Convert.FromBase64String serialized
                use ms = new System.IO.MemoryStream(bytes)
                use cs = new System.IO.Compression.DeflateStream(ms,System.IO.Compression.CompressionMode.Decompress)
                use br = new System.IO.BinaryReader(cs)
                // read size count
                let cnt = br.ReadInt32()
                let sizes = Array.init cnt (fun _ -> br.ReadInt32()) // read the sizes
                let network = Network.Create (fun _ -> br.ReadSingle()) sizes
                br.Close()
                network



    let dotProduct (x:float32[]) (M:float32[,]) (r:float32[]) =         //As the weights are already strucutured to handle the bias node in the output by ignoring it
        if x.Length < M.GetLength(1) || r.Length < M.GetLength(0) then
            failwithf "Can't dot x[%d] by M[%d,%d] to make r[%d] " x.Length (M.GetLength(0)) (M.GetLength(1)) r.Length
        let width,height = M.GetLength(1), M.GetLength(0)
        for j = 0 to height-1 do // we don't propagate to the bias
            let mutable sum = 0.f
            for i = 0 to width-1 do
                sum <- sum + x.[i]*M.[j,i]
            r.[j] <- sum

    let computeLogistic x =
            let v = 1. / (1. + (System.Math.Exp(-(float x))))
            float32 v

    let logistic length (x:float32[]) (r:float32[]) =                   //tells us how many elements of x to apply logistics to
        if r.Length < length || x.Length < length then
            failwithf "r[%d] is too short for x[%d]" r.Length x.Length
        for i = 0 to length-1 do
            let x' = x.[i]
            let v = 1. / (1. + (System.Math.Exp(-(float x'))))
            r.[i] <- float32 v

    let inverseLogistics length (x:float32[]) (r:float32[]) =
        for i = 0 to length-1 do
            let x' = float x.[i]
            let v = -System.Math.Log((1./x') - 1.)
            r.[i] <- float32 v


    let outputDeltas (outputs:float32[]) (expected:float32[]) (deltas:float32[]) =  //here we have things set up so we needn't worry about the bias nodes
        for i = 0 to expected.Length-1 do
            let o = outputs.[i]
            let t = expected.[i]
            deltas.[i] <- (o-t)*o*(1.f-o)       //(output - target)*output*(1-output)
    let innerDeltas (weights:float32[,]) (inputs:float32[]) (outputDeltas:float32[]) (deltas:float32[]) =   //this also deals with the bias node by nature of the weights' shape
        let width, height = weights.GetLength(1), weights.GetLength(0)
        for j = 0 to width-1 do
            let mutable sum = 0.f
            for l = 0 to height-1 do
                let weight = weights.[l,j]
                sum <- outputDeltas.[l]*weight + sum
            deltas.[j] <- sum*inputs.[j]*(1.f-inputs.[j])
           
    let updateWeights learningRate (weights:float32[,]) (inputs:float32[]) (outputDeltas:float32[]) =   //since the weights' structure size is preset we don't need to worry about the bias node
        let width, height = weights.GetLength(1), weights.GetLength(0)
        for j = 0 to height-1 do
            for i = 0 to width-1 do
                let weight = weights.[j,i]
                let delta = -learningRate*inputs.[i]*outputDeltas.[j]
                weights.[j,i] <- weight + delta

    let feedForward (network: Network) =
        for j = 0 to network.connections.Length-1 do
            let cm = network.connections.[j]
            let i = cm.inputLayer.nodes
            let o = cm.outputLayer.nodes
            let w = cm.weights
            dotProduct i w o
            // don't compute the logistic of the bias node
            logistic (o.Length-1) o o                       //make sure we don't apply logistics to the bias fn
        network.outputLayer.nodes

    let backprop learningRate (network: Network) =
        let outputLayer = network.outputLayer
        outputDeltas outputLayer.nodes network.expectedOutputs outputLayer.deltas
        for j = network.connections.Length-1 downto 1 do
            let connectionMatrix = network.connections.[j]
            let weights = connectionMatrix.weights
            let inLayer = connectionMatrix.inputLayer
            let outlayer = connectionMatrix.outputLayer
            innerDeltas weights inLayer.nodes outlayer.deltas inLayer.deltas
            updateWeights learningRate weights inLayer.nodes outlayer.deltas
        let connectionMatrix = network.connections.[0]
        updateWeights learningRate connectionMatrix.weights connectionMatrix.inputLayer.nodes connectionMatrix.outputLayer.deltas
        
    //// LOSS FNS
    //The array versions are sets up to only compare the number of nodes shared in common (because we aren't checking against the bias node)
    let distanceSquared (x:float32) (x':float32) = let d = x-x' in d*d  //literally what is says on the tin can
    let distanceSquaredArray (x:float32[]) (x':float32[]) =             //done over an entire array
        let mutable sum = 0.f
        let limit = min x.Length x'.Length
        for i = 0 to limit-1 do
            let d = x.[i] - x'.[i]
            sum <- sum + d*d
        sum
    let mseArray (x:float32[]) (x':float32[]) =                         //the AVERAGE distance squared
        let mutable sum = 0.f
        let limit = min x.Length x'.Length
        for i = 0 to limit-1 do
            let d = x.[i] - x'.[i]
            sum <- sum + d*d
        sum/(float32 limit)
    ////
    // here we are doing the training
    let train learningRate (network:Network) (trainingSet:seq<float32[]*float32[]>) (lossFunction:float32[] -> float32[] -> float32) =
        trainingSet                                 //tale the training set
        |> Seq.map (fun (i,e) ->                    //map each element to:
            network.LoadInput i                         //load the input (from the trainingset's input) into network
            network.LoadExpectedOutput e                //load the expected output (from the trainingset's output) into network
            let pred = feedForward network              //feed forward storing the resulting error into pred (for testing
            backprop learningRate network               //apply back prop
            lossFunction e pred                         //finally returning the resultant error of the loss function
        )
        |> Seq.average                              //get the average
    let check printErrors (network:Network) (validationSet:seq<float32[]*float32[]>) (lossFunction:float32[] -> float32[] -> float32) =
        validationSet                               //map the validation set
        |> Seq.map (fun (i,e) ->                    //to the same thing as above
            network.LoadInput i
            network.LoadExpectedOutput e
            let pred = feedForward network
            let loss = lossFunction e pred          //⋁⋁⋁ Print the resulting error
            if printErrors then
                printfn "i:%A pred: %A e: %A loss: %f" i pred e loss
            loss                                    //return the loss
        )
        |> Seq.average


    let testData() =
        // XOR
        [
            [|0.f;0.f|],[|1.f;0.f|]
            [|0.f;1.f|],[|0.f;1.f|]
            [|1.f;0.f|],[|0.f;1.f|]
            [|1.f;1.f|],[|1.f;0.f|]
        ]

    //remember that this is what the auto encoder is doing:
    //x -> NN -> x'
    //but since we're applying logistic, the outputs (at least) are always between 0 to 1. 
    //therefore you need (on regression at least) to scale the datasets

    //here we are creating some small-scale data to test on (2d boxes of points
    let testData2() =
        let rand = System.Random()
        let rand_m4_4() = rand.NextDouble() * 8. - 4.
        let region1 = (-3.5,-1.,3.,-0.25)
        let region2 = (-1.25,0.5,0.25,-2.)
        let region3 = (1.5,3.,2.5,0.)
        let regions = [region1; region2; region3]
        let insideRegion (x,y) (xl,xr,yt,yb) = xl <= x && x <= xr && yb <= y && y <= yt
        let insideRegions (x,y) = regions |> Seq.exists (insideRegion (x,y))
        let data =
            Seq.initInfinite (fun _ -> rand_m4_4(), rand_m4_4() )
            |> Seq.filter insideRegions
            |> Seq.take 200
            |> Seq.map (fun (x,y) -> float32 x, float32 y)
            |> Seq.toArray

        // remember we have to scale our data sets so that they fit into a region of zero to 1 so that the logistics based activation function can be used to decode the original values back
        // to do this we will preprocess the input data using the logistics function

        data
        |> Seq.map (fun (x,y) -> let i = [|computeLogistic x; computeLogistic y|] in i, i)  //this line with i being the input and output is the only difference between an AE and a normal backprop NN
        |> Seq.toArray
        //|> Seq.map (fun (x,y) -> sprintf "%f\t%f" x.[0] x.[1])
        //|> String.concat "\n"
        //|> toClipboard

    let testData3() =
        let rand = System.Random()
        let rand_m4_4() = rand.NextDouble() * 8. - 4.
        let f x = 2.3*x + 0.4
        let insideCurve (x,y) =
            let y' = f x
            if y = 0. then y = y'
            else
                abs((y-y')/y) < 0.01
        let data =
            Seq.initInfinite (fun _ -> rand_m4_4(), rand_m4_4() )
            |> Seq.filter insideCurve
            |> Seq.take 200
            |> Seq.map (fun (x,y) -> float32 x, float32 y)
            |> Seq.toArray

        // remember we have to scale our data sets so that they fit into a region of zero to 1 so that the logistics based activation function can be used to decode the original values back
        // to do this we will preprocess the input data using the logistics function

        data
        |> Seq.map (fun (x,y) -> let i = [|computeLogistic x; computeLogistic y|] in i, i, [|x;y|])  //this line with i being the input and output is the only difference between an AE and a normal backprop NN
        |> Seq.toArray

    let copyTo2D (a:float32[,]) (b:float32[,]) =
        for j = 0 to a.GetLength(0)-1 do
            for i = 0 to a.GetLength(1)-1 do
                b.[j,i] <- a.[j,i]

    let make1lvlSAE trainingCount learningRate inputLayerSize featureCount outputLayerSize trainingSetOriginal = 
        let sw = System.Diagnostics.Stopwatch.StartNew()
        let classifications = trainingSetOriginal |> Seq.map snd |> Seq.toArray
        let trainingSet = trainingSetOriginal |> Seq.map fst |> Seq.map (fun x -> x, x) |> Seq.toArray
        let lvl1 = Network.Create rand_uniform_1m_1 [|inputLayerSize;featureCount;inputLayerSize|]                              // Auto-encoder Level 1
        let lvlo = Network.Create rand_uniform_1m_1 [|featureCount;outputLayerSize|]                                   // Auto-encoder Level 3

        // Train the level 1 auto-encoder (1 auto-encoder layer)
        printfn "Training Level 1"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate lvl1 trainingSet distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss

        let trainingSet' =                                                                     // Training set to feed into the next level
            trainingSet
            |> Seq.mapi (fun n (i,_) ->                                                         // Map the values of the reduced set
                let pred = feedForward lvl1                                                     // Predicted values of level 2
                let r = Array.copy pred 
                r,classifications.[n]
            )

        printfn "Training Level Prediction"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate lvlo trainingSet' distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss

        let stackedAutoEncoder = 
            let cm1 = lvl1.connections.[0]                                                      // Connection matrix for Level 1 (goes to lvl2)
            let cmo = lvlo.connections.[0]                                                      // Connection matrix for Level O (goes to output)
            let cmo = {cmo with inputLayer = cm1.outputLayer;}                                  // Connect 3's CM with 2's output layer
            let network =                                                                       // Create the network
                {
                    connections = [|cm1;cmo|]                                                   // Connect 1's CM, 2's CM, and 3's CM
                    expectedOutputs = Array.zeroCreate cmo.outputLayer.Length                       // Create array of expected outputs
                }
            network.expectedOutputs.[network.expectedOutputs.Length-1] <- 1.f
            network

        printfn "Fine Turning SAE"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate stackedAutoEncoder trainingSetOriginal distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss
        sw.Stop()
        let elapsedTime = sw.Elapsed.TotalSeconds
        printfn "ElapsedTime: %fs" elapsedTime
        stackedAutoEncoder

    let make2lvlSAE trainingCount learningRate inputLayerSize featureCount1 featureCount2 outputLayerSize trainingSetOriginal = 
        let sw = System.Diagnostics.Stopwatch.StartNew()
        let classifications = trainingSetOriginal |> Seq.map snd |> Seq.toArray
        let trainingSet = trainingSetOriginal |> Seq.map fst |> Seq.map (fun x -> x, x) |> Seq.toArray
        let lvl1 = Network.Create rand_uniform_1m_1 [|inputLayerSize;featureCount1;inputLayerSize|]                              // Auto-encoder Level 1
        let lvl2 = Network.Create rand_uniform_1m_1 [|featureCount1;featureCount2;featureCount1|]                              // Auto-encoder Level 1
        let lvlo = Network.Create rand_uniform_1m_1 [|featureCount2;outputLayerSize|]                                   // Auto-encoder Level 3

        // Train the level 1 auto-encoder (1 auto-encoder layer)
        printfn "Training Level 1"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate lvl1 trainingSet distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss

        let trainingSet' =                                                                     // Training set to feed into the next level
            trainingSet
            |> Seq.mapi (fun n (i,_) ->                                                         // Map the values of the reduced set
                let pred = feedForward lvl1                                                     // Predicted values of level 2
                let r = Array.copy pred 
                r,r
            )

        // Train the level 2 auto-encoder (2 auto-encoder layer)
        printfn "Training Level 2"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate lvl2 trainingSet' distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss

        let trainingSet'' =                                                                     // Training set to feed into the next level
            trainingSet'
            |> Seq.mapi (fun n (i,_) ->                                                         // Map the values of the reduced set
                let pred = feedForward lvl2                                                     // Predicted values of level 2
                let r = Array.copy pred 
                r,classifications.[n]
            )

        printfn "Training Level Prediction"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate lvlo trainingSet'' distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss

        let stackedAutoEncoder = 
            let cm1 = lvl1.connections.[0]                                                      // Connection matrix for Level 1 (goes to lvl2)
            let cm2 = lvl2.connections.[0]                                                      // Connection matrix for Level 1 (goes to lvl2)
            let cmo = lvlo.connections.[0]                                                      // Connection matrix for Level O (goes to output)
            let cm2 = {cm2 with inputLayer = cm1.outputLayer;}                                  // Connect 3's CM with 2's output layer
            let cmo = {cmo with inputLayer = cm2.outputLayer;}                                  // Connect 3's CM with 2's output layer
            let network =                                                                       // Create the network
                {
                    connections = [|cm1;cm2;cmo|]                                                   // Connect 1's CM, 2's CM, and 3's CM
                    expectedOutputs = Array.zeroCreate cmo.outputLayer.Length                       // Create array of expected outputs
                }
            network.expectedOutputs.[network.expectedOutputs.Length-1] <- 1.f
            network

        printfn "Fine Turning SAE"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate stackedAutoEncoder trainingSetOriginal distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss
        sw.Stop()
        let elapsedTime = sw.Elapsed.TotalSeconds
        printfn "ElapsedTime: %fs" elapsedTime
        stackedAutoEncoder

    let make3lvlSAE trainingCount learningRate inputLayerSize featureCount1 featureCount2 featureCount3 outputLayerSize trainingSetOriginal = 
        let sw = System.Diagnostics.Stopwatch.StartNew()
        let classifications = trainingSetOriginal |> Seq.map snd |> Seq.toArray
        let trainingSet = trainingSetOriginal |> Seq.map fst |> Seq.map (fun x -> x, x) |> Seq.toArray
        let lvl1 = Network.Create rand_uniform_1m_1 [|inputLayerSize;featureCount1;inputLayerSize|]                              // Auto-encoder Level 1
        let lvl2 = Network.Create rand_uniform_1m_1 [|featureCount1;featureCount2;featureCount1|]                              // Auto-encoder Level 1
        let lvl3 = Network.Create rand_uniform_1m_1 [|featureCount2;featureCount3;featureCount2|]                              // Auto-encoder Level 1
        let lvlo = Network.Create rand_uniform_1m_1 [|featureCount3;outputLayerSize|]                                   // Auto-encoder Level 3

        // Train the level 1 auto-encoder (1 auto-encoder layer)
        printfn "Training Level 1"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate lvl1 trainingSet distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss

        let trainingSet' =                                                                     // Training set to feed into the next level
            trainingSet
            |> Seq.mapi (fun n (i,_) ->                                                         // Map the values of the reduced set
                let pred = feedForward lvl1                                                     // Predicted values of level 2
                let r = Array.copy pred 
                r,r
            )

        // Train the level 2 auto-encoder (2 auto-encoder layer)
        printfn "Training Level 2"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate lvl2 trainingSet' distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss

        let trainingSet'' =                                                                     // Training set to feed into the next level
            trainingSet'
            |> Seq.mapi (fun n (i,_) ->                                                         // Map the values of the reduced set
                let pred = feedForward lvl2                                                     // Predicted values of level 2
                let r = Array.copy pred 
                r,r
            )

        // Train the level 3 auto-encoder (3 auto-encoder layer)
        printfn "Training Level 3"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate lvl3 trainingSet'' distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss

        let trainingSet''' =                                                                     // Training set to feed into the next level
            trainingSet''
            |> Seq.mapi (fun n (i,_) ->                                                         // Map the values of the reduced set
                let pred = feedForward lvl3                                                     // Predicted values of level 2
                let r = Array.copy pred 
                r,classifications.[n]
            )

        printfn "Training Level Prediction"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate lvlo trainingSet''' distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss

        let stackedAutoEncoder = 
            let cm1 = lvl1.connections.[0]                                                      // Connection matrix for Level 1 (goes to lvl2)
            let cm2 = lvl2.connections.[0]                                                      // Connection matrix for Level 1 (goes to lvl2)
            let cm3 = lvl3.connections.[0]                                                      // Connection matrix for Level 1 (goes to lvl2)
            let cmo = lvlo.connections.[0]                                                      // Connection matrix for Level O (goes to output)
            let cm2 = {cm2 with inputLayer = cm1.outputLayer;}                                  // Connect 3's CM with 2's output layer
            let cm3 = {cm3 with inputLayer = cm2.outputLayer;}                                  // Connect 3's CM with 2's output layer
            let cmo = {cmo with inputLayer = cm3.outputLayer;}                                  // Connect 3's CM with 2's output layer
            let network =                                                                       // Create the network
                {
                    connections = [|cm1;cm2;cm3;cmo|]                                                   // Connect 1's CM, 2's CM, and 3's CM
                    expectedOutputs = Array.zeroCreate cmo.outputLayer.Length                       // Create array of expected outputs
                }
            network.expectedOutputs.[network.expectedOutputs.Length-1] <- 1.f
            network

        printfn "Fine Turning SAE"
        for i = 0 to trainingCount do
            let avgLoss = train learningRate stackedAutoEncoder trainingSetOriginal distanceSquaredArray
            if i%1000 = 0 then
                printfn "%d: %f" i avgLoss
        sw.Stop()
        let elapsedTime = sw.Elapsed.TotalSeconds
        printfn "ElapsedTime: %fs" elapsedTime
        stackedAutoEncoder


module Main =
    open Autoencoder
    open Datasets
    [<EntryPoint>]
    let main argv =
        System.Environment.CurrentDirectory <- __SOURCE_DIRECTORY__
        let dsmd1 = (fullDataset @"..\Data\abalone.data" (Some 0) None 2. true false) //filename classIndex regressionIndex pValue isCommaSeperated hasHeader
        let dsmd2 = (fullDataset @"..\Data\car.data" (Some 6) None 2. true false)
        let dsmd3 = (fullDataset @"..\Data\forestfires.csv" None (Some 12) 2. true true)
        let dsmd4 = (fullDataset @"..\Data\machine.data" None (Some 9) 2. true false )
        let dsmd5 = (fullDataset @"..\Data\segmentation.data" (Some 0) None 2. true true)
        let dsmd6 = (fullDataset @"..\Data\winequality-red.csv" None (Some 9) 2. false true)
        let dsmd7 = (fullDataset @"..\Data\winequality-white.csv" None (Some 11) 2. false true)

        let testSAEWithFold (makeSAE:_ -> Network) dsmd =            
            let folds = generateFolds dsmd
            let mse =
                folds
                |> Seq.mapi (fun fold (trainingSet,validationSet) ->
                    let sae = makeSAE trainingSet
                    let saeErr = check sae validationSet distanceSquaredArray
                    printfn "Fold [%d] error: %f" fold saeErr
                    saeErr
                )
                |> Seq.average
            printfn "MSE: %f" mse
    
            

        do
            dsmd1
            |> testSAEWithFold (make1lvlSAE 2000 1.f 8 5 3)
        
        0