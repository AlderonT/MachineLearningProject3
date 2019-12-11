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

// Open modules from local directory
open Types
open Extensions


// DATASETS MODULE
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
module Datasets =

    // Function to get a dataset from a file
    let fetchTrainingSet filePath isCommaSeperated hasHeader =
         System.IO.File.ReadAllLines(filePath)                                          // Read back a set of lines from the file 
         |> Seq.map (fun v -> v.Trim())                                                 // Trim the sequence
         |> Seq.filter (System.String.IsNullOrWhiteSpace >> not)                        // Filter out and remove white space
         |> Seq.filter (fun line ->                                                     // Filter out each line
             if isCommaSeperated && line.StartsWith(";") then false                     // Is it separated by commas or semicolons?
             else true
             )   
         |> (if hasHeader then Seq.skip 1 else id)                                      // Separate headers from data
         |> Seq.map (fun line -> line.Split(if isCommaSeperated then ',' else ';') |> Array.map (fun value -> value.Trim() |> System.String.Intern))    // Get an array of elements from the comma seperated fields, make sure that any white space is removed.
    
    // Logistic function for normalization
    let logistic (x:float32) = (1./(1.+System.Math.Exp(float -x) ))|>float32    

    // Function to get the full dataset
    let fullDataset filename (classIndex:int option) (regressionIndex : int option) (pValue:float) isCommaSeperated hasHeader= 
        let classIndex, regressionIndex =                                               // Is it classification or regression?
            match classIndex,regressionIndex with                                       // Match based on user's decision
            | None,None     -> -1,-1
            | None,Some a   -> -1,a 
            | Some a,None   -> a,-1
            | Some a,Some b -> a,b
        let dataSet = fetchTrainingSet filename isCommaSeperated hasHeader              // Get the training set
      
        // Grab the columns of the data file
        let columns = dataSet|> Seq.transpose|> Seq.toArray 

        // Sort between real and categorical indices
        let realIndexes, categoricalIndexes = 
            columns                                                                     // Map out the columns
            |>Seq.mapi (fun i c -> i,c)                                                 // Get the index and value
            |>Seq.filter (fun (i,_) -> i<>regressionIndex && i<> classIndex)            // Make sure the value does not equal the classification or regression index
            |>Seq.map (fun (i,c) ->                                                     // Grab the value
          
                i,
                (c
                    |> Seq.exists (fun v ->                                             // Check if the data value exists
                    v
                    |> System.Double.tryParse                                           // Is this index linked to a real value?
                    |> Option.isNone                                                    // Is this index linked to a categorical value?
                    )
                )
            )
            |>Seq.toArray                                                               // Store sequence of indices as array
            |>Array.partition snd                                                       // Partition out the array
            |>(fun (c,r) -> (r|> Seq.map fst |>Set.ofSeq),(c|>Seq.map fst |>Set.ofSeq)) // Map out the sequence of categorical and real values
      
        // Function to handle categorical values in a dataset
        let categoricalValues = 
            dataSet 
            |> Seq.collect (fun row -> row|>Seq.mapi (fun i value-> i,value.ToLowerInvariant()))    // value.ToLowerInvariant() forces the strings to all be lowercase
            |> Seq.filter (fst >> categoricalIndexes.Contains)                                      // Filter by categorical indices
            |> Seq.distinct                                                                         // Eliminates duplicate values
            |> Seq.groupBy fst                                                                      // Order from smallest to largest
            |> Seq.map (fun (catIdx,s)->                                                            // Map out the categorical values
                let values = 
                    s 
                    |> Seq.map snd                                                                  // Get the value of s
                    |> Seq.sort                                                                     // Sort by key
                    |> Seq.mapi (fun n v -> (v,n))                                                  // Map values
                    |> Map.ofSeq                                                                    // New map
                catIdx,values
            )
            |> Map.ofSeq                                                                            // New map

        // Function to get categorical node indices
        let categoricalNodeIndices = 
            categoricalValues                                                                       // Take the categorical values
            |> Seq.map (function KeyValue(k,v) -> k,v)                                              // Map the key values
            |> Seq.sortBy fst                                                                       // Sort by key
            |> Seq.mapFold (fun idx (k,v) ->                                                        
                let r = Array.init v.Count ((+) idx)                                                // Count through the values by index
                r,(idx+v.Count)
            ) 0 
            |> fst                                                                                  // Grab the sorted values
            |> Seq.toArray                                                                          // Send to array

        // Function to grab classification values
        let classificationValues =
            dataSet 
            |> Seq.collect (fun row -> row|>Seq.mapi (fun i value-> i,value.ToLowerInvariant()))    // value.ToLowerInvariant() forces the strings to all be lowercase
            |> Seq.filter (fst >> ((=) classIndex))                                                 // checks if the index is equal to the class index
            |> Seq.map snd                                                                          // Map the second value in the tuple
            |> Seq.distinct                                                                         // Remove duplicate values
            |> Seq.sort                                                                             // Sort the sequence by value
            |> Seq.toArray                                                                          // Convert to array

        // Function to set point metadata
        let metadata:DataSetMetadata = 
            { new DataSetMetadata with
                member _.getRealAttributeNodeIndex idx = if idx > realIndexes.Count then failwithf "Index %d is outside of range of real attributes!" idx else idx  // Real attribute node index
                member _.getCategoricalAttributeNodeIndices idx = categoricalNodeIndices.[idx]                                                                      // Categorical attribute node index
                member _.inputNodeCount = realIndexes.Count+(categoricalNodeIndices|> Seq.sumBy (fun x -> x.Length))                                                // Input node count
                member _.outputNodeCount = if regressionIndex <> -1 then 1 else classificationValues.Length                                                         // Output node count
                member _.getClassByIndex idx = if idx<classificationValues.Length then classificationValues.[idx] else "UNKNOWN"                                    // Class by Index
                member _.fillExpectedOutput point expectedOutputs =                                                                                                 // Expected Output
                    if regressionIndex<> -1 then expectedOutputs.[0] <- (logistic(point.regressionValue.Value|>float32) )
                    else    
                        for i = 0 to classificationValues.Length-1 do                                                                                               // Iterate through classification values and assign expected values
                            if point.cls.Value.ToLowerInvariant() = classificationValues.[i] then expectedOutputs.[i] <- 1.f
                            else expectedOutputs.[i] <- 0.f
                member _.isClassification = if regressionIndex <> -1 then false else true                                                                           // Set as classification or regression
            }

        // Function to set up a dataset
        let dataSet = 
            dataSet                                                                                                         
            |> Seq.map (fun p -> 
                {
                    cls = match classIndex with | -1 -> None | i -> Some p.[i]                                                  // Match class index
                    regressionValue = match regressionIndex with | -1 -> None | i -> (p.[i] |> System.Double.tryParse)          // Needs to be able to parse ints into floats
                    realAttributes = p |> Seq.filterWithIndex (fun i a -> realIndexes.Contains i) |>Seq.map System.Double.Parse |>Seq.map (fun x -> x|>float32)|> Seq.toArray
                    categoricalAttributes =                                                                                     // Take the categorical attributes
                        p 
                        |> Seq.chooseWithIndex (fun i a ->                                                                      // Select from index
                            match categoricalValues.TryFind i with                                                              // Match categorical values
                            | None -> None                                                                                      // If none, leave it be
                            | Some values -> values.TryFind a                                                                   // If they exist, find an a value to match up   
                            )
                        |> Seq.toArray                                                                                          // Convert sequence to array
                    metadata = metadata                                                                                         // Set metadata
                }
            ) |> Seq.toArray                                                                                                    // Convert sequence to array
        dataSet,metadata                                                                                                        // Return dataset and metadata

    // Function to set the input layer for a point
    let setInputLayerForPoint (n:Network) (p:Point) =
        let inputLayer = n.layers.[0]                                                                   // Grab the input layer
        for i = inputLayer.nodeCount to inputLayer.nodes.Length-1 do                                    // Iterate
            inputLayer.nodes.[i] <- 0.f                                                                 // Set nodes to 0
        p.realAttributes                                                                                // Grab the real attributes
        |> Seq.iteri (fun idx attributeValue ->                                                         // Iterate
            let nidx = p.metadata.getRealAttributeNodeIndex idx                                         // Get index
            inputLayer.nodes.[nidx] <- attributeValue                                                   // Grab attribute value
        )
        p.categoricalAttributes                                                                         // Grab the categorical attributes
        |> Seq.iteri (fun idx attributeValue ->                                                         // Iterate
            let nidxs = p.metadata.getCategoricalAttributeNodeIndices idx                               // Get index
            nidxs |> Seq.iteri (fun i nidx ->                                                           // Grab attribute value ...
                inputLayer.nodes.[nidx] <- if i = attributeValue then 1.f else 0.f                      // ... if unable, set as 0
            )
        )

    // Function to get the input layer for a point
    let getInputLayerFromPoint (p:Point) =
        let inputLayer = Array.zeroCreate (p.metadata.inputNodeCount)                               // Create an input layer
        p.realAttributes                                                                            // Take the real attributes of the point
        |> Seq.iteri (fun idx attributeValue ->                                                     // Iterate through the real attributes
            let nidx = p.metadata.getRealAttributeNodeIndex idx                                     // Get index   
            inputLayer.[nidx] <- logistic attributeValue                                            // Run logistic
        )
        p.categoricalAttributes                                                                     // Take the categorical attributes of the point
        |> Seq.iteri (fun idx attributeValue ->                                                     // Iterate through the categorical attributes
            let nidxs = p.metadata.getCategoricalAttributeNodeIndices idx                           // Get index
            nidxs |> Seq.iteri (fun i nidx ->
                inputLayer.[nidx] <- if i = attributeValue then 1.f else 0.f                        // Assign attribute values
            )
        )
        inputLayer

    // Function to set the input layer for a point
    let getOutputLayerFromPoint (p:Point) =
        let outputLayer = Array.zeroCreate (p.metadata.outputNodeCount)                             // Create a new array
        p.metadata.fillExpectedOutput p outputLayer                                                 // Fill in the expected output
        outputLayer                                                                                 // Return the output layer

    // Function for random K folds for K-Fold Cross Validation
    let getRandomFolds k (dataSet:'a seq) =                                                         // k is the number of slices dataset is the unsliced dataset
        let rnd = System.Random()                                                                   // Init randomnumbergenerator
        let data = ResizeArray(dataSet)                                                             // Convert our dataset to a resizable array
        let getRandomElement() =                                                                    // Get a random element out of data
            if data.Count <= 0 then None                                                            // If our data is empty return nothing
            else
                let idx = rnd.Next(0,data.Count)                                                    // Get a random index between 0 and |data|
                let e = data.[idx]                                                                  // Get the element e from idx
                data.RemoveAt(idx) |> ignore                                                        // Remove the element e from data
                Some e                                                                              // Return e
        let folds = Array.init k (fun _ -> Seq.empty)                                               // Resultant folds array init as an empty seq
        let rec generate  j =                                                                       // Recursively generate an array that increments on j (think like a while loop)
            match getRandomElement() with                                                           // Match the random element with:
            | None -> folds                                                                         // If there is nothing there then return folds
            | Some e ->                                                                             // If there is something there
                let s = folds.[j%k]                                                                 // Get the (j%k)th fold  in the array
                folds.[j%k] <- seq { yield! s; yield e }                                            // Create a new seqence containing the old sequence (at j%k) and the new element e, and put it back into slot (j%k)
                generate (j+1)                                                                      // Increment j and run again
        generate 0                                                                                  // Calls the generate function

    // Function to get a training set from a dataset
    let getTrainingSet (dsdm:Point[]*_) =
        let dataSet,_ = dsdm                                                                        // Set up dataset
        dataSet
        |> Seq.map (fun p ->                                                                        // Map the points
            let inData = getInputLayerFromPoint p                                                   // Set as input layer
            let expectedData = getOutputLayerFromPoint p                                            // Get expected values
            inData, expectedData                                                                    // Set as tuple
        )
        |> Seq.toArray                                                                              // Set as array

    // Function to generate folds for a dataset (10-Fold Crossover)
    let generateFolds dsdm =
        let dataSet = getTrainingSet dsdm                                                                               // Get training set
        let folds = getRandomFolds 10 dataSet                                                                           // Generate 10 random folds
        folds
        |> Seq.mapi (fun i v ->                                                                                         // Map each fold
            let trainingSet = folds |> Seq.filterWithIndex (fun i' _ -> i <> i') |> Seq.collect id |> Seq.toArray       // Assign training set
            let validationSet = v |> Seq.toArray                                                                        // Assign validation set 
            trainingSet, validationSet                                                                                  // Return as tuple
        )
        |> Seq.toArray                                                                                                  // Convert sequence to array
    
   
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
// END OF CODE