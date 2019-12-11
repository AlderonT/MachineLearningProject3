namespace SAE_NN
module Datasets =

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

    let generateFoldInputsAndOutputs (dataSet:Point seq) =
        let generateTrainingSetInputs (dataSet:Point seq) = 
            dataSet
            |> Seq.map(fun p -> getInputLayerFromPoint p) |> Seq.toArray
    
        let generateTrainingSetOutputs (dataSet:Point seq) =
            dataSet
            |> Seq.map(fun p -> getOutputLayerFromPoint p) |> Seq.toArray
        getRandomFolds 10 dataSet 
        |> Array.map (fun fo -> generateTrainingSetInputs fo,generateTrainingSetOutputs fo)
    
   