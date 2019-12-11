namespace SAE_NN
module rec Types =

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
        abstract member isClassification                    : bool                  //stores if this dataset is classification

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
    

    
