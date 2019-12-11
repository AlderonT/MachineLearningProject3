//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
//
//  CSCI 447 - Machine Learning
//  Extra Credit Assignment, Fall 2019
//  Chris Major, Farshina Nazrul-Shimim, Tysen Radovich, Allen Simpson
//
//  Implementation of a stacked auto-encoder neural network
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
                let e = sizes.GetEnumerator()                                               // We are creating an enumerator enumerating over sizes
                if e.MoveNext() |> not then                                                 // If size < 1 ... 
                    failwithf "ERROR: Need at least two sizes specified!"                   // ... fail
                let inputLayer = Layer.Create e.Current                                     // Initiate a layer using our enumerator's current element
                let rec loop inLayer (connections:ResizeArray<_>) =                         // Rec do:
                    if e.MoveNext() |> not then connections.ToArray()                       // If you can't move on, return the finished Array
                    else                                                                    // Else ...
                        let outLayer = Layer.Create e.Current                               // ... create the output layer
                        let cm = {                                                          // Create an instance of the connection matrix
                            inputLayer = inLayer                                            // Assign input layer
                            outputLayer = outLayer                                          // Assign output layer
                            
                            // Our initial weights are determined here: the size of the weights matrix takes into account that hte output layer should not write over the bias node.
                            weights = Array2D.init (outLayer.Length-1) inLayer.Length (fun _ _ -> nodeValue())  
                        }
                        connections.Add(cm)                                                 // Insert the connection matrix into connections
                        loop outLayer connections                                           // Now the next input layer is the current output layer
                let connections = loop inputLayer (ResizeArray())                           // Get the value from this recursive call and stick it in connectsion
                if connections.Length = 0 then                                              // If our thing is too small (size < 2) ....
                    failwithf "ERROR: Need at least two sizes specified!"                   // ... fail
                {                                                                           // Return our connections and our expected outputs in a Network object
                    connections = connections
                    expectedOutputs = Array.zeroCreate connections.[connections.Length-1].outputLayer.Length
                }

            // Low-level save function to copy-paste networks for testing
            member this.Save() =                                                                                       
                use ms = new System.IO.MemoryStream()                                                                   // Memory stream
                use cs = new System.IO.Compression.DeflateStream(ms,System.IO.Compression.CompressionLevel.Optimal)     // Stream compressor
                use bw = new System.IO.BinaryWriter(cs)                                                                 // Binary writer
                let sizes =                                                                                             // Get the sizes of the input, hidden layers, and outputlayer
                    seq {                                                                                               // Save as seq ...
                        yield!                                                      
                            this.connections                                                                            // ... containing the connecctions
                            |> Seq.map (fun cm -> cm.inputLayer.Length-1)                                               // Subtract one to get the original non-biased lengths for the layers
                        yield this.outputLayer.Length-1                                                                 // Containing our output layer without the bias layer
                    }
                    |> Seq.toArray                                                                                      // Return as an array
                
                bw.Write(sizes.Length)                                                                                  // Write the sizes out first
                sizes
                |> Seq.iter (bw.Write)
                this.connections                                                                                        // Writing the connections in 1st to last row-major order to the memory stream (same direction as construction)
                |> Seq.iter (fun cm ->
                    Array2D.iter (fun x -> bw.Write(x:float32)) cm.weights                                              // Now write out the elements
                )
                bw.Flush()                                                                                              // Flush out the binary writer
                bw.Close()                                                                                              // Close the binary writer
                let bytes = ms.ToArray()
                System.Convert.ToBase64String bytes                                                                     // Return the memory stream as a string in base 64

            // Static method to load by going backwards and producing a network
            static member Load(serialized:string) =                                                                     
                let bytes = System.Convert.FromBase64String serialized                                                  // Convert to Base 64
                use ms = new System.IO.MemoryStream(bytes)                                                              // New memory stream        
                use cs = new System.IO.Compression.DeflateStream(ms,System.IO.Compression.CompressionMode.Decompress)   // New compression stream
                use br = new System.IO.BinaryReader(cs)                                                                 // New binary writer
                let cnt = br.ReadInt32()
                let sizes = Array.init cnt (fun _ -> br.ReadInt32())                                                    // Read the sizes
                let network = Network.Create (fun _ -> br.ReadSingle()) sizes                                           // Create new network from loaded memory
                br.Close()                                                                                              // Close the binary writer
                network                                                                                                 // Return the network                                                                                


    // GENERAL FUNCTIONS
    //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

    // Function to calculate the dot product of x * M = r
    let dotProduct (x:float32[]) (M:float32[,]) (r:float32[]) = 
        if x.Length < M.GetLength(1) || r.Length < M.GetLength(0) then
            failwithf "ERROR: Can't dot x[%d] by M[%d,%d] to make r[%d]!" x.Length (M.GetLength(0)) (M.GetLength(1)) r.Length
        let width,height = M.GetLength(1), M.GetLength(0)                                                                       // Get M width and height
        for j = 0 to height-1 do                                                                                                // Don't propagate to the bias
            let mutable sum = 0.f                                                                                               // Create sum value
            for i = 0 to width-1 do                                                                                             // Iterate through M width
                sum <- sum + x.[i]*M.[j,i]                                                                                      // Perform dot product 
            r.[j] <- sum                                                                                                        // Store sum to output vector

    // Function to compute the logistic
    let computeLogistic x =
            let v = 1. / (1. + (System.Math.Exp(-(float x))))                                                           // Compute the logistic
            float32 v                                                                                                   // Return result as a float

    // Function to compute the logistic over an array
    let logistic length (x:float32[]) (r:float32[]) =                    
        if r.Length < length || x.Length < length then                                                                  // If r and x are too small ...
            failwithf "ERROR: r[%d] is too short for x[%d]!" r.Length x.Length                                          // ... error!
        for i = 0 to length-1 do                                                                                        // Iterate through the arrays
            let x' = x.[i]                                                                                              // Get the x value
            let v = 1. / (1. + (System.Math.Exp(-(float x'))))                                                          // Calculate the logistic function
            r.[i] <- float32 v                                                                                          // Store to the output vector

    // Function to compute the inverse logistic
    let inverseLogistics length (x:float32[]) (r:float32[]) =
        for i = 0 to length-1 do                                                                                        // Iterate through the arrays
            let x' = float x.[i]                                                                                        // Get the x value
            let v = -System.Math.Log((1./x') - 1.)                                                                      // Calculate the inverse logistic function
            r.[i] <- float32 v                                                                                          // Store to the output vector

    // Function to calculate the delta values of the output vector
    let outputDeltas (outputs:float32[]) (expected:float32[]) (deltas:float32[]) = 
        for i = 0 to expected.Length-1 do                                                                               // Iterate through the expected length
            let o = outputs.[i]                                                                                         // Get the predicted output value
            let t = expected.[i]                                                                                        // Get the expected output value
            deltas.[i] <- (o-t)*o*(1.f-o)                                                                               // Calculate the difference

    // Function to calculate the delta values of the input vector  
    let innerDeltas (weights:float32[,]) (inputs:float32[]) (outputDeltas:float32[]) (deltas:float32[]) =  
        let width, height = weights.GetLength(1), weights.GetLength(0)                                                  // Get weight matrix width and height
        for j = 0 to width-1 do                                                                                         // Iterate through matrix width
            let mutable sum = 0.f                                                                                       // Initialize mutable sum
            for l = 0 to height-1 do                                                                                    // Iterate through matrix height
                let weight = weights.[l,j]                                                                              // Grab the weight value
                sum <- outputDeltas.[l]*weight + sum                                                                    // Calculate the inner delta value
            deltas.[j] <- sum*inputs.[j]*(1.f-inputs.[j])                                                               // Store to delta array value
           
    // Function to update the weights of the weight matrix
    let updateWeights learningRate (weights:float32[,]) (inputs:float32[]) (outputDeltas:float32[]) =   
        let width, height = weights.GetLength(1), weights.GetLength(0)                                                  // Get weight matrix width and height
        for j = 0 to height-1 do                                                                                        // Iterate through matrix height
            for i = 0 to width-1 do                                                                                     // Iterate through matrix width
                let weight = weights.[j,i]                                                                              // Grab the weight value
                let delta = -learningRate*inputs.[i]*outputDeltas.[j]                                                   // Calculate the delta value
                weights.[j,i] <- weight + delta                                                                         // Update the weight with the delta


    // NEURAL NETWORK FUNCTIONS
    //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

    // Function to run a feedforward neural network
    let feedForward (network: Network) =
        for j = 0 to network.connections.Length-1 do                                                                    // Iterate through the connection matrix
            let cm = network.connections.[j]                                                                            // Grab a connection at index j
            let i = cm.inputLayer.nodes                                                                                 // Get the input nodes
            let o = cm.outputLayer.nodes                                                                                // Get the output nodes
            let w = cm.weights                                                                                          // Get the weights
            dotProduct i w o                                                                                            // Calculate the dot product
            logistic (o.Length-1) o o                                                                                   // Don't apply logistics to the bias fn
        network.outputLayer.nodes                                                                                       // Return the outputs

    // Function to perform backpropogation
    let backprop learningRate (network: Network) =
        let outputLayer = network.outputLayer                                                                                       // Get the output layer of the network
        outputDeltas outputLayer.nodes network.expectedOutputs outputLayer.deltas                                                   // Calculate the output delta error
        for j = network.connections.Length-1 downto 1 do                                                                            // Iterate through the connection matrix
            let connectionMatrix = network.connections.[j]                                                                          // Grab a connection
            let weights = connectionMatrix.weights                                                                                  // Grab the weights
            let inLayer = connectionMatrix.inputLayer                                                                               // Grab the input layer
            let outlayer = connectionMatrix.outputLayer                                                                             // Grab the output layer
            innerDeltas weights inLayer.nodes outlayer.deltas inLayer.deltas                                                        // Calculate the inner delta errors
            updateWeights learningRate weights inLayer.nodes outlayer.deltas                                                        // Update the weights
        let connectionMatrix = network.connections.[0]                                                                              // Set a connection matrix based on the initial network
        updateWeights learningRate connectionMatrix.weights connectionMatrix.inputLayer.nodes connectionMatrix.outputLayer.deltas   // Update the weights
        

    // LOSS FUNCTIONS
    //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

    // Function to calculate the square distance
    let distanceSquared (x:float32) (x':float32) = let d = x-x' in d*d  

    // Function to calculate the square distance over an entire array
    let distanceSquaredArray (x:float32[]) (x':float32[]) =             
        let mutable sum = 0.f                                                               // Create a mutable sum value                                          
        let limit = min x.Length x'.Length                                                  // Create limit based on smallest array dimension
        for i = 0 to limit-1 do                                                             // Iterate
            let d = x.[i] - x'.[i]                                                          // Get the difference value
            sum <- sum + d*d                                                                // Update the sum
        sum                                                                                 // Return the value

    // Function to calculate the mean square error
    let mseArray (x:float32[]) (x':float32[]) =                         
        let mutable sum = 0.f                                                               // Create a mutable sum value    
        let limit = min x.Length x'.Length                                                  // Create limit based on smallest array dimension
        for i = 0 to limit-1 do                                                             // Iterate
            let d = x.[i] - x'.[i]                                                          // Get the difference value
            sum <- sum + d*d                                                                // Update the sum
        sum/(float32 limit)                                                                 // Return the value

    // Function to conduct network training
    let train learningRate (network:Network) (trainingSet:seq<float32[]*float32[]>) (lossFunction:float32[] -> float32[] -> float32) =
        trainingSet                                                                         // Take the training set
        |> Seq.map (fun (i,e) ->                                                            // Map each element to:
            network.LoadInput i                                                             // Load the input (from the trainingset's input) into network
            network.LoadExpectedOutput e                                                    // Load the expected output (from the trainingset's output) into network
            let pred = feedForward network                                                  // Feed forward storing the resulting error into pred (for testing
            backprop learningRate network                                                   // Apply back prop
            lossFunction e pred                                                             // Return the resultant error of the loss function
        )
        |> Seq.average                                                                      // Get the average

    // Function to calculate (and report) the error of the network being run
    let check printErrors (network:Network) (validationSet:seq<float32[]*float32[]>) (lossFunction:float32[] -> float32[] -> float32) =
        validationSet                                                                       // Map the validation set ...
        |> Seq.map (fun (i,e) ->                                                            // ... to the same thing as above
            network.LoadInput i                                                             // Load the input
            network.LoadExpectedOutput e                                                    // Load the expected output
            let pred = feedForward network                                                  // Load the predictions from the feedforward neural network
            let loss = lossFunction e pred                                                  // Calculate the loss
            if printErrors then                                                             // If we want to print errors ...                                            
                printfn "i:%A pred: %A e: %A loss: %f" i pred e loss                        // ... print the errors
            loss                                                                            // Return the loss
        )
        |> Seq.average                                                                      // Average and return


    // Function for mapping 2D spaces for testing (we wanted to plot some 2D representations earlier on in development)
    let copyTo2D (a:float32[,]) (b:float32[,]) =
        for j = 0 to a.GetLength(0)-1 do                                                    // Iterate
            for i = 0 to a.GetLength(1)-1 do                                                // Iterate
                b.[j,i] <- a.[j,i]                                                          // Copy


    // STACKED AUTO-ENCODER FUNCTIONS
    //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

    // Function to run a Level 1, 1 Auto-Encoder layer stacked auto-encoder
    let make1lvlSAE trainingCount learningRate inputLayerSize featureCount outputLayerSize trainingSetOriginal = 
        let sw = System.Diagnostics.Stopwatch.StartNew()                                                                // Start stopwatch for time diagnostics
        let classifications = trainingSetOriginal |> Seq.map snd |> Seq.toArray                                         // Classifier for training data
        let trainingSet = trainingSetOriginal |> Seq.map fst |> Seq.map (fun x -> x, x) |> Seq.toArray                  // Training set
        let lvl1 = Network.Create rand_uniform_1m_1 [|inputLayerSize;featureCount;inputLayerSize|]                      // Auto-encoder first level
        let lvlo = Network.Create rand_uniform_1m_1 [|featureCount;outputLayerSize|]                                    // Auto-encoder output level

        // Train the level 1 auto-encoder (1 auto-encoder layer)
        printfn "Training Level 1 ..."                                                                                  // Print message
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times
            let avgLoss = train learningRate lvl1 trainingSet distanceSquaredArray                                      // Train the first auto-encoder layer
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss

        let trainingSet' =                                                                                              // Training set to feed into the next level
            trainingSet
            |> Seq.mapi (fun n (i,_) ->                                                                                 // Map the values of the reduced set
                let pred = feedForward lvl1                                                                             // Predicted values of level 2
                let r = Array.copy lvl1.featureLayer.nodes                                                              // Copy nodes
                r,classifications.[n]                                                                                   // Return to training set
            ) |> Seq.toArray

        // Predict the training level (1 auto-encoder layer)
        printfn "Training Level Prediction"                                                                             // Print message
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times
            let avgLoss = train learningRate lvlo trainingSet' distanceSquaredArray                                     // Train the output auto-encoder layer
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss

        // Connect the stacked auto-encoder
        let stackedAutoEncoder = 
            let cm1 = lvl1.connections.[0]                                                                              // Connection matrix for Level 1 (goes to lvl2)
            let cmo = lvlo.connections.[0]                                                                              // Connection matrix for Level O (goes to output)
            let cmo = {cmo with inputLayer = cm1.outputLayer;}                                                          // Connect 3's CM with 2's output layer
            let network =                                                                                               // Create the network
                {
                    connections = [|cm1;cmo|]                                                                           // Connect 1's CM, 2's CM, and 3's CM
                    expectedOutputs = Array.zeroCreate cmo.outputLayer.Length                                           // Create array of expected outputs
                }
            network.expectedOutputs.[network.expectedOutputs.Length-1] <- 1.f                                           // Set the bias nodes
            network                                                                                                     // Return the network

        // Fine-tuning stage
        printfn "Fine Turning SAE"                                                                                      // Print message
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times
            let avgLoss = train learningRate stackedAutoEncoder trainingSetOriginal distanceSquaredArray                // Train the overall auto-encoder
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss              
        sw.Stop()                                                                                                       // Stop the stopwatch
        let elapsedTime = sw.Elapsed.TotalSeconds
        //printfn "ElapsedTime: %fs" elapsedTime
        stackedAutoEncoder                                                                                              // Return the stacked autoencoder

    // Function to run a Level 2, 2 Auto-Encoder layer stacked auto-encoder
    let make2lvlSAE trainingCount learningRate inputLayerSize featureCount1 featureCount2 outputLayerSize trainingSetOriginal = 
        let sw = System.Diagnostics.Stopwatch.StartNew()                                                                // Start stopwatch for time diagnostics
        let classifications = trainingSetOriginal |> Seq.map snd |> Seq.toArray                                         // Classifier for training data
        let trainingSet = trainingSetOriginal |> Seq.map fst |> Seq.map (fun x -> x, x) |> Seq.toArray                  // Training set
        let lvl1 = Network.Create rand_uniform_1m_1 [|inputLayerSize;featureCount1;inputLayerSize|]                     // Auto-encoder first level
        let lvl2 = Network.Create rand_uniform_1m_1 [|featureCount1;featureCount2;featureCount1|]                       // Auto-encoder second level
        let lvlo = Network.Create rand_uniform_1m_1 [|featureCount2;outputLayerSize|]                                   // Auto-encoder output level

        // Train the level 1 auto-encoder (1 auto-encoder layer)
        printfn "Training Level 1"                                                                                      // Print message
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times
            let avgLoss = train learningRate lvl1 trainingSet distanceSquaredArray                                      // Train the first auto-encoder layer
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss

        let trainingSet' =                                                                                              // Training set to feed into the next level
            trainingSet
            |> Seq.mapi (fun n (i,_) ->                                                                                 // Map the values of the reduced set
                let pred = feedForward lvl1                                                                             // Predicted values of level 2
                let r = Array.copy lvl1.featureLayer.nodes 
                r,r
            ) |> Seq.toArray

        // Train the level 2 auto-encoder (2 auto-encoder layer)
        printfn "Training Level 2"                                                                                      // Print message   
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times
            let avgLoss = train learningRate lvl2 trainingSet' distanceSquaredArray                                     // Train the second auto-encoder layer
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss

        let trainingSet'' =                                                                                             // Training set to feed into the next level
            trainingSet'
            |> Seq.mapi (fun n (i,_) ->                                                                                 // Map the values of the reduced set
                let pred = feedForward lvl2                                                                             // Predicted values of level 2
                let r = Array.copy lvl2.featureLayer.nodes                                                              // Copy nodes
                r,classifications.[n]                                                                                   // Return to training set
            ) |> Seq.toArray

        printfn "Training Level Prediction"                                                                             // Print message
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times
            let avgLoss = train learningRate lvlo trainingSet'' distanceSquaredArray                                    // Train the output auto-encoder layer
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss

        let stackedAutoEncoder = 
            let cm1 = lvl1.connections.[0]                                                                              // Connection matrix for Level 1 (goes to lvl2)
            let cm2 = lvl2.connections.[0]                                                                              // Connection matrix for Level 1 (goes to lvl2)
            let cmo = lvlo.connections.[0]                                                                              // Connection matrix for Level O (goes to output)
            let cm2 = {cm2 with inputLayer = cm1.outputLayer;}                                                          // Connect 3's CM with 2's output layer
            let cmo = {cmo with inputLayer = cm2.outputLayer;}                                                          // Connect 3's CM with 2's output layer
            let network =                                                                                               // Create the network
                {
                    connections = [|cm1;cm2;cmo|]                                                                       // Connect 1's CM, 2's CM, and 3's CM
                    expectedOutputs = Array.zeroCreate cmo.outputLayer.Length                                           // Create array of expected outputs
                }
            network.expectedOutputs.[network.expectedOutputs.Length-1] <- 1.f                                           // Set the bias nodes
            network                                                                                                     // Return the network

        printfn "Fine Turning SAE"                                                                                      // Print message
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times
            let avgLoss = train learningRate stackedAutoEncoder trainingSetOriginal distanceSquaredArray                // Train the overall auto-encoder
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss
        sw.Stop()                                                                                                       // Stop the stopwatch
        let elapsedTime = sw.Elapsed.TotalSeconds
        //printfn "ElapsedTime: %fs" elapsedTime
        stackedAutoEncoder                                                                                              // Return the stacked autoencoder

    // Function to run a Level 3, 3 Auto-Encoder layer stacked auto-encoder    
    let make3lvlSAE trainingCount learningRate inputLayerSize featureCount1 featureCount2 featureCount3 outputLayerSize trainingSetOriginal = 
        let sw = System.Diagnostics.Stopwatch.StartNew()                                                                // Start stopwatch for time diagnostics
        let classifications = trainingSetOriginal |> Seq.map snd |> Seq.toArray                                         // Classifier for training data
        let trainingSet = trainingSetOriginal |> Seq.map fst |> Seq.map (fun x -> x, x) |> Seq.toArray                  // Training set
        let lvl1 = Network.Create rand_uniform_1m_1 [|inputLayerSize;featureCount1;inputLayerSize|]                     // Auto-encoder first level
        let lvl2 = Network.Create rand_uniform_1m_1 [|featureCount1;featureCount2;featureCount1|]                       // Auto-encoder second level
        let lvl3 = Network.Create rand_uniform_1m_1 [|featureCount2;featureCount3;featureCount2|]                       // Auto-encoder third level
        let lvlo = Network.Create rand_uniform_1m_1 [|featureCount3;outputLayerSize|]                                   // Auto-encoder output level

        // Train the level 1 auto-encoder (1 auto-encoder layer)
        printfn "Training Level 1"                                                                                      // Print message
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times
            let avgLoss = train learningRate lvl1 trainingSet distanceSquaredArray                                      // Train the first auto-encoder layer
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss

        let trainingSet' =                                                                                              // Training set to feed into the next level
            trainingSet
            |> Seq.mapi (fun n (i,_) ->                                                                                 // Map the values of the reduced set
                let pred = feedForward lvl1                                                                             // Predicted values of level 2
                let r = Array.copy lvl1.featureLayer.nodes 
                r,r
            ) |> Seq.toArray

        // Train the level 2 auto-encoder (2 auto-encoder layer)
        printfn "Training Level 2"                                                                                      // Print message
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times
            let avgLoss = train learningRate lvl2 trainingSet' distanceSquaredArray                                     // Train the second auto-encoder layer
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss

        let trainingSet'' =                                                                                             // Training set to feed into the next level
            trainingSet'
            |> Seq.mapi (fun n (i,_) ->                                                                                 // Map the values of the reduced set
                let pred = feedForward lvl2                                                                             // Predicted values of level 2
                let r = Array.copy lvl2.featureLayer.nodes
                r,r
            ) |> Seq.toArray

        // Train the level 3 auto-encoder (3 auto-encoder layer)
        printfn "Training Level 3"                                                                                      // Print message
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times      
            let avgLoss = train learningRate lvl3 trainingSet'' distanceSquaredArray                                    // Train the third auto-encoder layer
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss

        let trainingSet''' =                                                                                            // Training set to feed into the next level
            trainingSet''
            |> Seq.mapi (fun n (i,_) ->                                                                                 // Map the values of the reduced set
                let pred = feedForward lvl3                                                                             // Predicted values of level 2
                let r = Array.copy lvl3.featureLayer.nodes 
                r,classifications.[n]
            ) |> Seq.toArray

        printfn "Training Level Prediction"                                                                             // Print message
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times
            let avgLoss = train learningRate lvlo trainingSet''' distanceSquaredArray                                   // Train the output auto-encoder layer
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss

        let stackedAutoEncoder = 
            let cm1 = lvl1.connections.[0]                                                                              // Connection matrix for Level 1 (goes to lvl2)
            let cm2 = lvl2.connections.[0]                                                                              // Connection matrix for Level 1 (goes to lvl2)
            let cm3 = lvl3.connections.[0]                                                                              // Connection matrix for Level 1 (goes to lvl2)
            let cmo = lvlo.connections.[0]                                                                              // Connection matrix for Level O (goes to output)
            let cm2 = {cm2 with inputLayer = cm1.outputLayer;}                                                          // Connect 3's CM with 2's output layer
            let cm3 = {cm3 with inputLayer = cm2.outputLayer;}                                                          // Connect 3's CM with 2's output layer
            let cmo = {cmo with inputLayer = cm3.outputLayer;}                                                          // Connect 3's CM with 2's output layer
            let network =                                                                                               // Create the network
                {
                    connections = [|cm1;cm2;cm3;cmo|]                                                                   // Connect 1's CM, 2's CM, and 3's CM
                    expectedOutputs = Array.zeroCreate cmo.outputLayer.Length                                           // Create array of expected outputs
                }
            network.expectedOutputs.[network.expectedOutputs.Length-1] <- 1.f                                           // Set the bias nodes
            network                                                                                                     // Return the network

        printfn "Fine Turning SAE"                                                                                      // Print message
        for i = 0 to trainingCount do                                                                                   // Iterate trainingCount times
            let avgLoss = train learningRate stackedAutoEncoder trainingSetOriginal distanceSquaredArray                // Train the overall auto-encoder 
            if i%1000 = 0 then                                                                                          // Print updates every 1000 iterations
                printfn "%d: %f" i avgLoss
        sw.Stop()
        let elapsedTime = sw.Elapsed.TotalSeconds                                                                       // Stop the stopwatch
        //printfn "ElapsedTime: %fs" elapsedTime
        stackedAutoEncoder                                                                                              // Return the stacked autoencoder


// MAIN MODULE
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

module Main =

    // Open Modules
    open Autoencoder
    open Datasets
    [<EntryPoint>]

    // Main function
    let main argv =
        //System.Environment.CurrentDirectory <- __SOURCE_DIRECTORY__

        // filename classIndex regressionIndex pValue isCommaSeperated hasHeader
        let dsmd1 = (fullDataset @"..\..\..\Data\abalone.data" (Some 0) None 2. true false)                       // Abalone
        let dsmd2 = (fullDataset @"..\..\..\Data\car.data" (Some 6) None 2. true false)                           // Car
        let dsmd3 = (fullDataset @"..\..\..\Data\forestfires.csv" None (Some 12) 2. true true)                    // Forest Fires
        let dsmd4 = (fullDataset @"..\..\..\Data\machine.data" None (Some 9) 2. true false )                      // Machine
        let dsmd5 = (fullDataset @"..\..\..\Data\segmentation.data" (Some 0) None 2. true true)                   // Segmentation
        let dsmd6 = (fullDataset @"..\..\..\Data\winequality-red.csv" None (Some 9) 2. false true)                // Wine Quality (Red)
        let dsmd7 = (fullDataset @"..\..\..\Data\winequality-white.csv" None (Some 11) 2. false true)             // Wine Quality (White)

        dsmd2 |> (fun (x,u) -> u.inputNodeCount,u.outputNodeCount)

        // Test the Stacked Auto-encoder
        let testSAEWithFold msg (makeSAE:_ -> Network) dsmd =            
            printfn "Processing SAE with %s" msg                                                            // Print message
            let sw = System.Diagnostics.Stopwatch.StartNew()                                                // Start stopwatch
            let folds = generateFolds dsmd                                                                  // Generate K-folds
            let mse =                                                                                       // Calculate mean square error for each fold
                folds
                |> Seq.take 1
                |> Seq.mapi (fun fold (trainingSet,validationSet) ->                                        // Map folds to training set and validation set
                    async {
                        let sae = makeSAE trainingSet                                                       // Make the SAE training set
                        let saeErr = check false sae validationSet distanceSquaredArray                     // Check SAE validation
                        printfn "Fold [%d] Error: %f" fold saeErr                                           // Print errors for fold
                        return saeErr                                                                       // Return SAE error
                    }
                )
                |> Async.Parallel                                                                           // Parallelism for speed
                |> Async.RunSynchronously
                |> Seq.average
            sw.Stop()                                                                                       // Stop the stopwatch
            let elapsedTime = sw.Elapsed.TotalSeconds                                                       // Total time run    
            printfn "ElapsedTime: %fs" elapsedTime                                                          // Print time run
            printfn "MSE: %f" mse                                                                           // Print MSE
    
            
        // Run through the 1, 2, and 3 auto-encoder layer configurations
        //do
        //    dsmd1
        //    |> testSAEWithFold "8 5 3" (make1lvlSAE 2000 1.f 8 5 3)                     // 1 auto-encoder layer
        //    dsmd1
        //    |> testSAEWithFold "8 5 4 3" (make2lvlSAE 2000 1.f 8 5 4 3)                   // 2 auto-encoder layers
        //    dsmd1
        //    |> testSAEWithFold "8 6 5 4 3" (make3lvlSAE 2000 1.f 8 6 5 4 3)                 // 3 auto-encoder layers
        
        do
            dsmd2
            |> testSAEWithFold "21 12 4" (make1lvlSAE 2000 1.f 21 12 4)                     // 1 auto-encoder layer
            dsmd2
            |> testSAEWithFold "21 12 4 4" (make2lvlSAE 2000 1.f 21 12 4 4)                   // 2 auto-encoder layers
            dsmd2
            |> testSAEWithFold "21 12 5 4 4" (make3lvlSAE 2000 1.f 21 12 5 4 4)                 // 3 auto-encoder layers

        0


//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
// END OF CODE