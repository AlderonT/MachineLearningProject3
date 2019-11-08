//--------------------------------------------------------------------------------------------------------------
//
//  CSCI 447 - Machine Learning
//  Assignment #3, Fall 2019
//  Chris Major, Farshina Nazrul-Shimim, Tysen Radovich, Allen Simpson
//
//  [DESCRIPTION]
//
//--------------------------------------------------------------------------------------------------------------

// MODULES
//--------------------------------------------------------------------------------------------------------------
namespace Project3

    // Load the file names from the same folder
    #load "FeedforwardNet.fsx"
    #load "RBFNet.fsx"
    #load "tools.fsx"
    
    // Open the modules
    open FeedforwardNet
    open RBFNet
    open Tools

    // Declare as a module
    module rec Assignment3 = 


// OBJECTS
//--------------------------------------------------------------------------------------------------------------
        
        // Create a Metadata object to distinguish real and categorical attributes by index
        type DataSetMetadata = 
            abstract member getRealAttributeNodeIndex           : int -> int            // Indices of the real attributes
            abstract member getCategoricalAttributeNodeIndices  : int -> int[]          // Indices of the categorical attributes
            abstract member inputNodeCount                      : int                   // number of input nodes
            abstract member outputNodeCount                     : int                   // number of output nodes
            abstract member getClassByIndex                     : int -> string         // get the class associated with this node's index
            abstract member fillExpectedOutput                  : Point -> float32[] -> unit    //assigned the expected output of Point to the float32[]

        // Create a Layer object to represent a layer within the neural network
        type Layer = {
            nodes                                   : float32[]                         // Sequence to make up vectors
            nodeCount                               : int                               // Number of nodes in the layer
            deltas                                  : float32[]                         // sacrificing space for speed
        }
        // Create a ConnectionMatrix object to represent the connection matrix within the neural network
        type ConnectionMatrix = {
            weights                                 : float32[]                         // Sequence of weights within the matrix
            inputLayer                              : Layer                             // Input layer
            outputLayer                             : Layer                             // Output layer
        }
        // Create a Network object to represent a neural network
        type Network = {
            layers                                  : Layer[]                           // Array of layers within the network
            connections                             : ConnectionMatrix[]                // Array of connections within the network
        }
            with 
                member this.outLayer = this.layers.[this.layers.Length-1]
                member this.inLayer = this.layers.[0]

        // Create a Point object to represent a point within the data
        type Point = {
            realAttributes                          : float32[]                         // the floating point values for the real points
            categoricalAttributes                   : int[]                             // the values for categorical attributes. distance will be discrete
            cls                                     : string option
            regressionValue                         : float option
            metadata                                : DataSetMetadata
        }

            // Method for the Point object (not entirely functional but simple enough for application)
            with 
                member this.distance p = //sqrt((Real distance)^2+(categorical distance)^2) //categorical distance = 1 if different value, or 0 if same value
                    (Seq.zip this.realAttributes p.realAttributes|> Seq.map (fun (a,b) -> a-b)|> Seq.sumBy (fun d -> d*d))
                    + (Seq.zip this.categoricalAttributes p.categoricalAttributes |> Seq.sumBy (fun (a,b)-> if a=b then 0.f else 1.f))
                    |>sqrt 
       

// FUNCTIONS
//--------------------------------------------------------------------------------------------------------------
       
        //// How to get a dataset from a file
        let fetchTrainingSet filePath isCommaSeperated hasHeader =
            System.IO.File.ReadAllLines(filePath)                           // this give you back a set of line from the file (replace with your directory)
            |> Seq.map (fun v -> v.Trim())                                  // trim the sequence
            |> Seq.filter (System.String.IsNullOrWhiteSpace >> not)         // filter out and remove white space
            |> Seq.filter (fun line ->                                      // take each line
                if isCommaSeperated && line.StartsWith(";") then false      // separate by commas or semicolons
                else true
                )   
            |> (if hasHeader then Seq.skip 1 else id)                       // separate headers from data
            |> Seq.map (fun line -> line.Split(if isCommaSeperated then ',' else ';') |> Array.map (fun value -> value.Trim() |> System.String.Intern)) // this give you an array of elements from the comma seperated fields. We trim to make sure that any white space is removed.
    
        // Write out functions
        
        ////GET THE DATASET
        let fullDataset filename (classIndex:int option) (regressionIndex : int option) (pValue:float) isCommaSeperated hasHeader= 
            let classIndex,regressionIndex = 
                match classIndex,regressionIndex with 
                | None,None     -> -1,-1
                | None,Some a   -> -1,a 
                | Some a,None   -> a,-1
                | Some a,Some b -> a,b
            let dataSet = fetchTrainingSet filename isCommaSeperated hasHeader
            
            ////Need to comment this!
            let columns = dataSet|> Seq.transpose|> Seq.toArray 
            let realIndexes,categoricalIndexes = 
                columns
                |>Seq.mapi (fun i c -> i,c)
                |>Seq.filter (fun (i,_) -> i<>regressionIndex && i<> classIndex)
                |>Seq.map (fun (i,c) ->
                
                    i,
                    (c
                     |> Seq.exists (fun v -> 
                        v
                        |>System.Double.tryParse 
                        |> Option.isNone
                        )
                    )
                )
                |>Seq.toArray
                |>Array.partition snd
                |>(fun (c,r) -> (r|> Seq.map fst |>Set.ofSeq),(c|>Seq.map fst |>Set.ofSeq))
            
            let categoricalValues = 
                dataSet 
                |> Seq.collect (fun row -> row|>Seq.mapi (fun i value-> i,value.ToLowerInvariant())) //value.ToLowerInvariant() forces the strings to all be lowercase
                |> Seq.filter (fst >> categoricalIndexes.Contains)
                |> Seq.distinct
                |> Seq.groupBy fst
                |> Seq.map (fun (catIdx,s)->
                    let values = 
                        s 
                        |> Seq.map snd
                        |> Seq.sort
                        |> Seq.mapi (fun n v -> (v,n))
                        |> Map.ofSeq
                    catIdx,values
                )
                |> Map.ofSeq

            let categoricalNodeIndices = 
                categoricalValues 
                |> Seq.map (function KeyValue(k,v)-> (k,Array.init v.Count id))
                |> Seq.sortBy fst 
                |> Seq.map snd 
                |> Seq.toArray

            let classificationValues =
                dataSet 
                |> Seq.collect (fun row -> row|>Seq.mapi (fun i value-> i,value.ToLowerInvariant())) //value.ToLowerInvariant() forces the strings to all be lowercase
                |> Seq.filter (fst >> ((=) classIndex)) //checks if the index is equal to the class index
                |> Seq.map snd
                |> Seq.distinct
                |> Seq.sort
                |> Seq.toArray                

            let metadata:DataSetMetadata = 
                { new DataSetMetadata with
                    member _.getRealAttributeNodeIndex idx = if idx > realIndexes.Count then failwithf "index %d is outside of range of real attributes" idx else idx 
                    member _.getCategoricalAttributeNodeIndices idx = categoricalNodeIndices.[idx]
                    member _.inputNodeCount = realIndexes.Count+(categoricalNodeIndices|> Seq.sumBy (fun x -> x.Length))
                    member _.outputNodeCount = if regressionIndex <> -1 then 1 else classificationValues.Length
                    member _.getClassByIndex idx = if idx<classificationValues.Length then classificationValues.[idx] else "UNKNOWN"
                    member _.fillExpectedOutput point expectedOutputs = 
                        if regressionIndex<>-1 then expectedOutputs.[0] = point.regressionValue
                        else    
                            for i = 0 to classificationValues.Length-1 do 
                                if point.cls.Value.ToLowerInvariant() = classificationValues.[i] then expectedOutputs.[i] <- 1.f
                                else expectedOutputs.[i] <- 0.f
                }
            let dataSet = 
                dataSet
                |> Seq.map (fun p -> 
                    {
                        cls = match classIndex with | -1 -> None | i -> Some p.[i]
                        regressionValue = match regressionIndex with | -1 -> None | i -> (p.[i] |> System.Double.tryParse) //Needs to be able to parse ints into floats
                        realAttributes = p |> Seq.filterWithIndex (fun i a -> realIndexes.Contains i) |>Seq.map System.Double.Parse |>Seq.map (fun x -> x|>float32)|> Seq.toArray
                        categoricalAttributes = 
                            p 
                            |> Seq.chooseWithIndex (fun i a -> 
                                match categoricalValues.TryFind i with
                                | None -> None 
                                | Some values -> values.TryFind a 
                                )
                            |> Seq.toArray
                        metadata = metadata
                    }
                ) |> Seq.toArray
            dataSet,metadata
        let setInputLayerForPoint (n:Network) (p:Point) =
            let inputLayer = n.layers.[0]
            for i = inputLayer.nodeCount to inputLayer.nodes.Length-1 do 
                inputLayer.nodes.[i] <- 0.f
            p.realAttributes 
            |> Seq.iteri (fun idx attributeValue -> 
                let nidx = p.metadata.getRealAttributeNodeIndex idx 
                inputLayer.nodes.[nidx] <- attributeValue 
            )
            p.categoricalAttributes 
            |> Seq.iteri (fun idx attributeValue -> 
                let nidxs = p.metadata.getCategoricalAttributeNodeIndices idx
                nidxs |> Seq.iter (fun nidx ->
                    inputLayer.nodes.[nidx] <- if nidx = attributeValue then 1.f else 0.f 
                )
            )

        let createNetwork (metadata:DataSetMetadata) hiddenLayerSizes =    
            let multipleOfFour i =  i+((4-(i%4))%4)
            let allocatedInputNodeCount = multipleOfFour metadata.inputNodeCount    //adjusting to make the input length a multiple of 4
            let allocatedOutputNodeCount = multipleOfFour metadata.outputNodeCount  //adjusting to make the input length a multiple of 4
            
            let layers = 
                seq {
                    yield {
                        nodes = Array.zeroCreate allocatedInputNodeCount 
                        nodeCount = metadata.inputNodeCount
                        deltas = Array.zeroCreate allocatedInputNodeCount 
                    } 
                    
                    yield! 
                        hiddenLayerSizes
                        |>Array.map (fun size ->
                            let allocatedSize = multipleOfFour size
                            {
                                nodes = Array.zeroCreate allocatedSize
                                nodeCount = size
                                deltas = Array.zeroCreate allocatedSize
                            }
                        )
                    
                    yield {
                        nodes = Array.zeroCreate allocatedOutputNodeCount
                        nodeCount = metadata.outputNodeCount
                        deltas = Array.zeroCreate allocatedOutputNodeCount
                    }

                }
                |>Seq.toArray

            let createConnectionMatrix (inLayer,outLayer) = 
                {
                    weights = Array.zeroCreate (inLayer.nodes.Length*outLayer.nodes.Length)
                    inputLayer = inLayer
                    outputLayer = outLayer
                }
            
            {
                layers = layers 
                connections = layers |> Seq.pairwise |> Seq.map createConnectionMatrix |> Seq.toArray
            }
        let initializeNetwork network = 
            let rand = System.Random()
            let initializeConnectionMatrix cMatrix = 
                for i = 0 to cMatrix.weights.Length-1 do 
                    cMatrix.weights.[i]<-rand.NextDouble()|>float32 //we can set these weights to be random values without tracking the phantom weights 
                                                                    //because everything will work so long as the phantom input nodes are set to 0, 
                                                                    //and the delta(phantom output nodes) are set to 0 on backprop 
            network.connections |> Seq.iter initializeConnectionMatrix

        let feedForward (metadata:DataSetMetadata) network point = 
            let logistic (x:float32) = (1./(1.+System.Math.Exp(float -x) ))|>float32    //Logistic Fn
            let outputLayer = network.layers.[network.layers.Length-1]                  //output layer def
            setInputLayerForPoint network point                                         //set the input layer to the point
            let runThroughConnection connection = 
                for j = 0 to connection.outputLayer.nodeCount-1 do
                    let mutable sum = 0.f
                    for i = 0 to connection.inputLayer.nodeCount-1 do 
                        let k = connection.inputLayer.nodes.Length * j+i 
                        sum <- sum + connection.weights.[k]*connection.inputLayer.nodes.[i]
                    if j < connection.outputLayer.nodeCount then 
                        connection.outputLayer.nodes.[j]<-logistic sum
                    else 
                        connection.outputLayer.nodes.[j]<- 0.f
            network.connections
            |>Seq.iter runThroughConnection
            outputLayer.nodes
            |> Seq.mapi (fun i v -> v,i)
            |> Seq.max 
            |> fun (v,i) -> v,metadata.getClassByIndex i
        
        let outputDeltas (outputs:float32[]) (expected:float32[]) (deltas:float32[]) =
            Seq.zip outputs expected
            |> Seq.iteri (fun i (o,t) ->
                deltas.[i] <- (o-t)*o*(1.f-o)       //(output - target)*output*(1-output)
            )
        let innerDeltas (weights:float32[]) (inputs:float32[]) (outputDeltas:float32[]) (deltas:float32[]) =
            for j = 0 to inputs.Length-1 do
                let mutable sum = 0.f
                for l = 0 to outputDeltas.Length-1 do
                    let jl = l*outputDeltas.Length+j
                    let weight = weights.[jl]
                    sum <- outputDeltas.[l]*weight + sum
                deltas.[j] <- sum*inputs.[j]*(1.f-inputs.[j])
           
        let updateWeights learningRate (weights:float32[]) (inputs:float32[]) (outputDeltas:float32[]) =
            for j = 0 to outputDeltas.Length-1 do
                for i = 0 to inputs.Length-1 do
                    let ij = j*outputDeltas.Length+i
                    let weight = weights.[ij]
                    let delta = -learningRate*inputs.[i]*outputDeltas.[j]
                    weights.[ij] <- weight + delta
        
        let computeError (network:Network) (expectedoutput:float32[])=
            let outLayer = network.outLayer
            let mutable errSum = 0.f
            for i = 0 to outLayer.nodeCount do 
                errSum <- let d = outLayer.nodes.[i] - expectedoutput.[i] in d*d+errSum
            errSum/2.f

        let backprop learningRate (network: Network) (expectedOutputs:float32[]) =
            let outputLayer = network.outLayer 
            outputDeltas outputLayer.nodes expectedOutputs outputLayer.deltas
            for j = network.connections.Length-1 to 1 do    
                let connectionMatrix = network.connections.[j]
                let inLayer = connectionMatrix.inputLayer
                let outlayer = connectionMatrix.outputLayer
                innerDeltas connectionMatrix.weights inLayer.nodes outlayer.deltas inLayer.deltas
                updateWeights learningRate connectionMatrix.weights inLayer.nodes outlayer.deltas
            let connectionMatrix = network.connections.[0]
            updateWeights learningRate connectionMatrix.weights connectionMatrix.inputLayer.nodes connectionMatrix.outputLayer.deltas
        
        let trainNetwork learningRate (metadata:DataSetMetadata) (network: Network) (trainingSet:Point[]) = 
            let expectedOutputs = Array.zeroCreate metadata.outputNodeCount
            trainingSet
            |> Seq.mapi (fun i p->
                metadata.fillExpectedOutput p expectedOutputs
                let activationValue,cls = feedForward metadata network p
                let totalErr = computeError network expectedOutputs
                printfn "Error for point %d: %f " i totalErr
                backprop learningRate network expectedOutputs
                totalErr
            )
            |> Seq.sum
            |> fun x-> x/(trainingSet.Length|> float32)
        
        let runNetwork (metadata:DataSetMetadata) (network: Network) (point:Point) =
            let expectedOutputs = Array.zeroCreate metadata.outputNodeCount
            metadata.fillExpectedOutput p expectedOutputs
            let activationValue,cls = feedForward metadata network point
            let err = computeError network expectedOutputs
            cls,activationValue,err 
            
        let trainNetworkToErr epsilon learningRate (metadata:DataSetMetadata) (network: Network) (trainingSet:Point[]) =
            let rec loop count=
                let err = trainNetwork  learningRate metadata network trainingSet
                printfn "Run %d: %f" count err
                if err<= epsilon then ()
                else loop (count+1)
            loop 0

// IMPLEMENTATIONS
//--------------------------------------------------------------------------------------------------------------
        do
            let ds1,metadata = (fullDataset @"D:\Fall2019\Machine Learning\MachineLearningProject3\Data\car.data" (Some 6) None 2. true false) //filename classIndex regressionIndex pValue isCommaSeperated hasHeader
            let network = createNetwork metadata [|10;10;10|]
            initializeNetwork network 
            feedForward metadata network ds1.[10]
            ds1|> Array.map (fun p -> 
                let cost,cls = feedForward metadata network p
                cost,cls,p.cls
            )
            |> Array.countBy(fun (_,b,c)-> b,c)

            network.layers.[network.layers.Length-1].nodes
           

//--------------------------------------------------------------------------------------------------------------
// END OF CODE