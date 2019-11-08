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
    module Assignment3 = 


// OBJECTS
//--------------------------------------------------------------------------------------------------------------
        
        // Create a Metadata object to distinguish real and categorical attributes by index
        type DataSetMetadata = 
            abstract member getRealAttributeNodeIndex           : int -> int            // Indices of the real attributes
            abstract member getCategoricalAttributeNodeIndices  : int -> int[]          // Indices of the categorical attributes
            abstract member inputNodeCount                      : int                   // number of input nodes
            abstract member outputNodeCount                     : int                   // number of output nodes
            abstract member getClassByIndex                     : int -> string         // get the class associated with this node's index

        // Create a Layer object to represent a layer within the neural network
        type Layer = {
            nodes                                   : float32[]                         // Sequence to make up vectors
            nodeCount                               : int                               // Number of nodes in the layer
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
       

        // Function to get the training set from a file
        //------------------------------------------------------------------------------------------------------
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
        

        // Function to get the full dataset from a file
        //------------------------------------------------------------------------------------------------------
        let fullDataset filename (classIndex:int option) (regressionIndex : int option) (pValue:float) isCommaSeperated hasHeader= 
            
            // Set up the class and regression indices
            let classIndex, regressionIndex = 
                match classIndex,regressionIndex with 
                | None, None     -> -1, -1
                | None, Some a   -> -1, a 
                | Some a, None   -> a, -1
                | Some a, Some b -> a, b

            // Fetch the training set from the file
            let dataSet = fetchTrainingSet filename isCommaSeperated hasHeader
            
            // Grab the columns
            let columns = dataSet |> Seq.transpose |> Seq.toArray 

            // Divide the real and categorical data indexes
            let realIndexes, categoricalIndexes = 
                
                // Run through the columns
                columns
                |> Seq.mapi (fun i c -> i,c)                                                // Map each column by indices
                |> Seq.filter (fun (i,_) -> i<>regressionIndex && i<> classIndex)           // Filter classification and regression indexs
                |> Seq.map (fun (i,c) ->                                                    // Map through each item
                
                    // Run through index i
                    i,
                    (c
                     |> Seq.exists (fun v ->                                                // If a value exists ...
                        v
                        |> System.Double.tryParse                                           // System try statement
                        |> Option.isNone                                                    // Parse data
                        )
                    )
                )
                |>Seq.toArray                                                               // Convert sequence
                |>Array.partition snd                                                       // Partition the sequence
                |>(fun (c,r) -> (r|> Seq.map fst |>Set.ofSeq),(c|>Seq.map fst |>Set.ofSeq)) // Re-map sequence
            
            // Grab the categorical values
            let categoricalValues = 

                // Run through the data set
                dataSet 
                |> Seq.collect (fun row -> row|>Seq.mapi (fun i value-> i,value.ToLowerInvariant()))    // value.ToLowerInvariant() forces the strings to all be lowercase
                |> Seq.filter (fst >> categoricalIndexes.Contains)                                      // Filter the features by categorical        
                |> Seq.distinct                                                                         // Distinguish these features    
                |> Seq.groupBy fst                                                                      // Group       
                |> Seq.map (fun (catIdx, s) ->                                                          // Map category indices        
                    
                    // Define the categorical values
                    let values =  
                            
                        // Run through s
                        s 
                        |> Seq.map snd                                                                  // Map the sequence
                        |> Seq.sort                                                                     // Sort by index
                        |> Seq.mapi (fun n v -> (v,n))                                                  // Map by index
                        |> Map.ofSeq                                                                    // Create map

                    // Tuple of values
                    catIdx, values                                                                       
                )
                |> Map.ofSeq                                                                            // Complete mapping

            // Grab the categorical node indices
            let categoricalNodeIndices = 

                // Run through the categorical values
                categoricalValues 
                |> Seq.map (function KeyValue(k, v)-> (k,Array.init v.Count id))                        // Map values in KeyValues of array and count values
                |> Seq.sortBy fst                                                                       // Sort through tuples
                |> Seq.map snd                                                                          // Map through sequence to tuple values
                |> Seq.toArray                                                                          // Convert to array   

            // Grab the classification values
            let classificationValues =

                // Run through the dataset
                dataSet 
                |> Seq.collect (fun row -> row|>Seq.mapi (fun i value-> i,value.ToLowerInvariant()))    // value.ToLowerInvariant() forces the strings to all be lowercase
                |> Seq.filter (fst >> ((=) classIndex))                                                 // Check if the index is equal to the class index
                |> Seq.map snd                                                                          // Map through sequence to tuple values
                |> Seq.distinct                                                                         // Remove any duplicates
                |> Seq.sort                                                                             // Sort the contents
                |> Seq.toArray                                                                          // Convert the sequence to an array        

            // Create a metadata object to contain the real and categorical data for each point
            let metadata:DataSetMetadata = 
                { new DataSetMetadata with
                    member _.getRealAttributeNodeIndex idx = if idx > realIndexes.Count then failwithf "index %d is outside of range of real attributes" idx else idx   // Real
                    member _.getCategoricalAttributeNodeIndices idx = categoricalNodeIndices.[idx]                                                                      // Categorical
                    member _.inputNodeCount = realIndexes.Count+(categoricalNodeIndices|> Seq.sumBy (fun x -> x.Length))                                                // Input Node Count
                    member _.outputNodeCount = if regressionIndex <> -1 then 1 else classificationValues.Length                                                         // Output Node Count
                    member _.getClassByIndex idx = if idx<classificationValues.Length then classificationValues.[idx] else "UNKNOWN"                                    // Class by Index
                }

            // Create a dataset value
            let dataSet = 

                // Run through the dataset
                dataSet
                |> Seq.map (fun p ->                                                                                        // Map the dataset to value p
                    {
                        cls = match classIndex with | -1 -> None | i -> Some p.[i]                                          // Match the class with the index value
                        regressionValue = match regressionIndex with | -1 -> None | i -> (p.[i] |> System.Double.tryParse)  // Needs to be able to parse ints into floats
                        realAttributes = p |> Seq.filterWithIndex (fun i a -> realIndexes.Contains i) |>Seq.map System.Double.Parse |>Seq.map (fun x -> x|>float32)|> Seq.toArray   // Parse real attributes into values
                        
                        // Run through the categorical attributes
                        categoricalAttributes = 

                            // Run through each categorical value
                            p 
                            |> Seq.chooseWithIndex (fun i a ->                  // Select from sequence by index
                                match categoricalValues.TryFind i with          // Find categorical values based on indices
                                | None -> None                                  // If none, leave alone
                                | Some values -> values.TryFind a               // If there is a value, try in sequence
                                )
                            |> Seq.toArray                                      // Convert to array

                        // Assign to the object metadata
                        metadata = metadata
                    }
                ) |> Seq.toArray

            // Return a tuple of the dataset and metadata
            dataSet, metadata


        // Function to set the input layer for a point
        //------------------------------------------------------------------------------------------------------
        let setInputLayerForPoint (n:Network) (p:Point) =
            let inputLayer = n.layers.[0]                                                       // Grab the first layer as input
            
            // Iterate through the layers                                                       
            for i = inputLayer.nodeCount to inputLayer.nodes.Length - 1 do                           
                inputLayer.nodes.[i] <- 0.f                                                     // Set to zeroes
            
            // Run through the real attributes of the point
            p.realAttributes 
            |> Seq.iteri (fun idx attributeValue ->                                             // Iterate
                let nidx = p.metadata.getRealAttributeNodeIndex idx                             // Grab the real attribute index from the metadata
                inputLayer.nodes.[nidx] <- attributeValue                                       // Assign attribute value to input node    
            )

            // Run through the categorical attributes of the point
            p.categoricalAttributes 
            |> Seq.iteri (fun idx attributeValue ->                                             // Iterate
                let nidxs = p.metadata.getCategoricalAttributeNodeIndices idx                   // Grab the categorical attribute index from the metadata
                nidxs |> Seq.iter (fun nidx ->
                    inputLayer.nodes.[nidx] <- if nidx = attributeValue then 1.f else 0.f       // Assign categorical value if it exists
                )
            )


        // Function to create a neural network
        //------------------------------------------------------------------------------------------------------        
        let createNetwork (metadata:DataSetMetadata) hiddenLayerSizes =    
            
            let multipleOfFour i =  i+((4-(i%4))%4)                                     // Create value wtih multiple of four    
            let allocatedInputNodeCount = multipleOfFour metadata.inputNodeCount        // Adjusting to make the input length a multiple of 4
            let allocatedOutputNodeCount = multipleOfFour metadata.outputNodeCount      // Adjusting to make the input length a multiple of 4
            
            // Craete a layers sequence
            let layers = 
                
                seq {
                    yield {                                                             // Add single item to sequence
                        nodes = Array.zeroCreate allocatedInputNodeCount                // Create empty array of nodes
                        nodeCount = metadata.inputNodeCount                             // Store number of nodes for input
                    } 
                    
                    yield!                                                              // Add all items to sequence
                        
                        // Run through the hidden layer size values
                        hiddenLayerSizes                                                                       
                        |> Array.map (fun size ->                                       // Map the hidden layer by size
                            let allocatedSize = multipleOfFour size                     // Define a variable of size
                            {
                                nodes = Array.zeroCreate allocatedSize                  // Create an empty array of nodes
                                nodeCount = size                                        // Set value of the node count
                            }
                        )
                    
                    yield {                                                             // Add single item to sequence
                        nodes = Array.zeroCreate allocatedOutputNodeCount               // Create empty array of nodes
                        nodeCount = metadata.outputNodeCount                            // Store number of nodes for output    
                    }

                }
                |>Seq.toArray                                                           // Convert sequence to array

            // Create a connection matrix with the input and output layers
            let createConnectionMatrix (inLayer, outLayer) = 
                {
                    weights = Array.zeroCreate (inLayer.nodes.Length*outLayer.nodes.Length)             // Create an array of zeroes for the weights
                    inputLayer = inLayer                                                                // Create the input layer
                    outputLayer = outLayer                                                              // Create the output layer
                }
            
            {
                layers = layers                                                                         // Create the intermediate layers
                connections = layers |> Seq.pairwise |> Seq.map createConnectionMatrix |> Seq.toArray   // Determine the number of layers
            }


        // Function to initialize the neural network
        //------------------------------------------------------------------------------------------------------
        let initializeNetwork network =
        
            // Reference random value generator
            let rand = System.Random()
            
            // Initialize the connection matrix
            let initializeConnectionMatrix cMatrix = 
                for i = 0 to cMatrix.weights.Length-1 do 
                    cMatrix.weights.[i]<-rand.NextDouble()|>float32                         // We can set these weights to be random values without tracking the phantom weights 
                                                                                            // because everything will work so long as the phantom input nodes are set to 0, 
                                                                                            // and the delta(phantom output nodes) are set to 0 on backprop 
            network.connections |> Seq.iter initializeConnectionMatrix


        // Function for the Feedforward Neural Network
        //------------------------------------------------------------------------------------------------------
        let feedForward (metadata:DataSetMetadata) network point = 
            
            let logistic (x:float32) = (1./(1.+System.Math.Exp(float -x) )) |> float32      // Logistic Function
            
            let outputLayer = network.layers.[network.layers.Length-1]                      // Output layer def
            
            setInputLayerForPoint network point                                             // Set the input layer to the point
            
            // Connect all of the points in the network
            let runThroughConnection connection = 

                // Iterate through the layers
                for j = 0 to connection.outputLayer.nodeCount - 1 do

                    // Create mutable value to hold summation
                    let mutable sum = 0.f

                    // Iterate through the input layer rows
                    for i = 0 to connection.inputLayer.nodeCount - 1 do 

                        // Iterate through the columns
                        let k = connection.inputLayer.nodes.Length * j + i 

                        // Add to the summation value
                        sum <- sum + connection.weights.[k] * connection.inputLayer.nodes.[i]

                    // Store the values in the output layer            
                    connection.outputLayer.nodes.[j]<-logistic sum
            
            // Return network connection values
            network.connections

            |>Seq.iter runThroughConnection                 // Iterate through the sequence
            
            // Run through the output layer nodes
            outputLayer.nodes           
            |> Seq.mapi (fun i v -> v,i)                    // Map each node by index
            |> Seq.max                                      // Grab the maximum
            |> fun (v,i) -> v, metadata.getClassByIndex i   // Return as the classification/regression value
        

        // Function for the Radial Basis Function
        //------------------------------------------------------------------------------------------------------
        let RBFNetwork (metadata:DataSetMetadata) network point = 
            
            let logistic (x:float32) = (1./(1.+System.Math.Exp(float -x) )) |> float32      // Logistic Function
            
            let outputLayer = network.layers.[network.layers.Length-1]                      // Output layer def
            
            setInputLayerForPoint network point                                             // Set the input layer to the point
            
            // Connect all of the points in the network
            let runThroughConnection connection = 

                // Iterate through the layers
                for j = 0 to connection.outputLayer.nodeCount - 1 do

                    // Create mutable value to hold summation
                    let mutable sum = 0.f

                    // Iterate through the input layer rows
                    for i = 0 to connection.inputLayer.nodeCount - 1 do 

                        // Iterate through the columns
                        let k = connection.inputLayer.nodes.Length * j + i 

                        // Add to the summation value
                        // [TODO] Modify with Gaussian function
                        //sum <- sum + connection.weights.[k] * connection.inputLayer.nodes.[i]

                    // Store the values in the output layer            
                    connection.outputLayer.nodes.[j]<-logistic sum
            
            // Return network connection values
            network.connections

            |>Seq.iter runThroughConnection                 // Iterate through the sequence
            
            // Run through the output layer nodes
            outputLayer.nodes           
            |> Seq.mapi (fun i v -> v,i)                    // Map each node by index
            |> Seq.max                                      // Grab the maximum
            |> fun (v,i) -> v, metadata.getClassByIndex i   // Return as the classification/regression value

// IMPLEMENTATIONS
////--------------------------------------------------------------------------------------------------------------
//        do
//            let ds1,metadata = (fullDataset @"D:\Fall2019\Machine Learning\MachineLearningProject3\Data\car.data" (Some 6) None 2. true false) //filename classIndex regressionIndex pValue isCommaSeperated hasHeader
//            let network = createNetwork metadata [|10;10;10|]
//            initializeNetwork network 
//            feedForward metadata network ds1.[10]
//            ds1|> Array.map (fun p -> 
//                let cost,cls = feedForward metadata network p
//                cost,cls,p.cls
//            )
//            |> Array.countBy(fun (_,b,c)-> b,c)

//            network.layers.[network.layers.Length-1].nodes
           

//--------------------------------------------------------------------------------------------------------------
// END OF CODE