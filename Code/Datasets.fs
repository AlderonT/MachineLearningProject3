namespace SAE_NN
open Types
open Extensions
module Datasets =

    
    // How to get a dataset from a file
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
            |> Seq.map (function KeyValue(k,v) -> k,v)
            |> Seq.sortBy fst
            |> Seq.mapFold (fun idx (k,v) ->
                let r = Array.init v.Count ((+) idx)
                r,(idx+v.Count)
            ) 0
            |> fst
            |> Seq.toArray

        let classificationValues =
            dataSet 
            |> Seq.collect (fun row -> row|>Seq.mapi (fun i value-> i,value.ToLowerInvariant())) //value.ToLowerInvariant() forces the strings to all be lowercase
            |> Seq.filter (fst >> ((=) classIndex)) //checks if the index is equal to the class index
            |> Seq.map snd
            |> Seq.distinct
            |> Seq.sort
            |> Seq.toArray                
        let logistic (x:float32) = (1./(1.+System.Math.Exp(float -x) ))|>float32    //Logistic Fn
        let metadata:DataSetMetadata = 
            { new DataSetMetadata with
                member _.getRealAttributeNodeIndex idx = if idx > realIndexes.Count then failwithf "index %d is outside of range of real attributes" idx else idx 
                member _.getCategoricalAttributeNodeIndices idx = categoricalNodeIndices.[idx]
                member _.inputNodeCount = realIndexes.Count+(categoricalNodeIndices|> Seq.sumBy (fun x -> x.Length))
                member _.outputNodeCount = if regressionIndex <> -1 then 1 else classificationValues.Length
                member _.getClassByIndex idx = if idx<classificationValues.Length then classificationValues.[idx] else "UNKNOWN"
                member _.fillExpectedOutput point expectedOutputs = 
                    if regressionIndex<> -1 then expectedOutputs.[0] <- (logistic(point.regressionValue.Value|>float32) )
                    else    
                        for i = 0 to classificationValues.Length-1 do 
                            if point.cls.Value.ToLowerInvariant() = classificationValues.[i] then expectedOutputs.[i] <- 1.f
                            else expectedOutputs.[i] <- 0.f
                member _.isClassification = if regressionIndex <> -1 then false else true
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
            nidxs |> Seq.iteri (fun i nidx ->
                inputLayer.nodes.[nidx] <- if i = attributeValue then 1.f else 0.f 
            )
        )

    let getInputLayerFromPoint (p:Point) =
        let inputLayer = Array.zeroCreate (p.metadata.inputNodeCount)
        p.realAttributes 
        |> Seq.iteri (fun idx attributeValue -> 
            let nidx = p.metadata.getRealAttributeNodeIndex idx 
            inputLayer.[nidx] <- attributeValue 
        )
        p.categoricalAttributes 
        |> Seq.iteri (fun idx attributeValue -> 
            let nidxs = p.metadata.getCategoricalAttributeNodeIndices idx
            nidxs |> Seq.iteri (fun i nidx ->
                inputLayer.[nidx] <- if i = attributeValue then 1.f else 0.f 
            )
        )
        inputLayer

    let getOutputLayerFromPoint (p:Point) =
        let outputLayer = Array.zeroCreate (p.metadata.outputNodeCount)
        p.metadata.fillExpectedOutput p outputLayer
        outputLayer


    let getRandomFolds k (dataSet:'a seq) = //k is the number of slices dataset is the unsliced dataset
        let rnd = System.Random()           //init randomnumbergenerator
        let data = ResizeArray(dataSet)     //convert our dataset to a resizable array
        let getRandomElement() =            //Get a random element out of data
            if data.Count <= 0 then None    //if our data is empty return nothing
            else
                let idx = rnd.Next(0,data.Count)    //get a random index between 0 and |data|
                let e = data.[idx]                  //get the element e from idx
                data.RemoveAt(idx) |> ignore        //remove the element e from data
                Some e                              //return e
        let folds = Array.init k (fun _ -> Seq.empty)       //resultant folds array init as an empty seq
        let rec generate  j =                               //recursively generate an array that increments on j (think like a while loop)
            match getRandomElement() with                   //match the random element with:
            | None -> folds                                 //if there is nothing there then return folds
            | Some e ->                                     // if there is something there
                let s = folds.[j%k]                         // get the (j%k)th fold  in the array
                folds.[j%k] <- seq { yield! s; yield e }    //create a new seqence containing the old sequence (at j%k) and the new element e, and put it back into slot (j%k)
                generate (j+1)                              //increment j and run again
        generate 0                                          //calls the generate function

    let getTrainingSet (dsdm:Point[]*_) =
        let dataSet,_ = dsdm
        dataSet
        |> Seq.map (fun p ->
            let inData = getInputLayerFromPoint p
            let expectedData = getOutputLayerFromPoint p
            inData, expectedData
        )
        |> Seq.toArray
    //let generateFoldInputsAndOutputs (dataSet:Point seq) =
    //    let generateTrainingSetInputs (dataSet:Point seq) = 
    //        dataSet
    //        |> Seq.map(fun p -> getInputLayerFromPoint p) |> Seq.toArray
    
    //    let generateTrainingSetOutputs (dataSet:Point seq) =
    //        dataSet
    //        |> Seq.map(fun p -> getOutputLayerFromPoint p) |> Seq.toArray
    //    getRandomFolds 10 dataSet 
    //    |> Array.map (fun fo -> generateTrainingSetInputs fo,generateTrainingSetOutputs fo)
    let generateFolds dsdm =
        let dataSet = getTrainingSet dsdm
        let folds = getRandomFolds 10 dataSet
        folds
        |> Seq.mapi (fun i v ->
            let trainingSet = folds |> Seq.filterWithIndex (fun i' _ -> i <> i') |> Seq.collect id |> Seq.toArray
            let validationSet = v
            trainingSet, validationSet
        )
        |> Seq.toArray
    
   