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

            let metadata:DataSetMetadata = 
                { new DataSetMetadata with
                    member _.getRealAttributeNodeIndex idx = if idx > realIndexes.Count then failwithf "index %d is outside of range of real attributes" idx else idx 
                    member _.getCategoricalAttributeNodeIndices idx = categoricalNodeIndices.[idx]
                }
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
        
        let setInputLayerForPoint (n:Network) (p:Point) =
            let inputLayer = n.layers.[0]
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
           

// IMPLEMENTATIONS
//--------------------------------------------------------------------------------------------------------------
        do
            let ds1 = (fullDataset @"D:\Fall2019\Machine Learning\Project 2\Data\abalone.data" (Some 0) None 2. true false) //filename classIndex regressionIndex pValue isCommaSeperated hasHeader

            let filename = @"C:\Users\chris\OneDrive\Documents\CSCI447\MachineLearningProject3\Data\car.data" 
            let classIndex = Some 6 
            let regressionIndex = None
            let pValue = 2.
            let isCommaSeperated,hasHeader = true,false
            ()
//--------------------------------------------------------------------------------------------------------------
// END OF CODE