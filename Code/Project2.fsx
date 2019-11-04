//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//
//  CSCI 447 - Machine Learning      
//  Assignment #2
//  Chris Major, Farshina Nazrul-Shimim, Tysen Radovich, Allen Simpson
//
//  Implementation and demonstration of K-nearest Neighbor (KNN), Edited Nearest Neighbor (ENN),
//  Condensed Nearest Neighbor (CNN), K-Means Regression, and K-Medoids algorithms.
//
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Load files for the module
#load "tools.fsx"
open Tools

// Module Declaration for runtime
module Project2 =


    // CLASSES
    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    type CategoricalAttributePercentages = Map<int,Map<string,Map<string,float>>>
    type RegressionAttributePercentages = Map<int,Map<string,float>>
    type AttributeDetails =
        {
            categoricalAttributePercentages: CategoricalAttributePercentages
            regressionAverages: RegressionAttributePercentages
        }
    type Point = 
        abstract member realAttributes: float[]
        abstract member categoricalAttributes: string[]
        abstract member cls : string option
        abstract member regressionValue : float option
        abstract member distance: p:Point -> attributeDetails:AttributeDetails-> float      //sqrt((Real distance)^2+(CategoricalClassification distance)^2+(CategoricalRegression distance)^2)
 
 
    // Class for a Classification process output
    type Classifier =
        abstract member classify: p:Point -> string option

    // Class for a Regression process output
    type Regresser =
        abstract member regress: p:Point -> float


    // FUNCTIONS
    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    //// How to get a dataset from a file
    let fetchTrainingSet filePath isCommaSeperated hasHeader =
        System.IO.File.ReadAllLines(filePath)                           // this give you back a set of line from the file (replace with your directory)
        |> Seq.map (fun v -> v.Trim())                                  // trim the sequence
        |> Seq.filter (System.String.IsNullOrWhiteSpace >> not)         // filter out and remove white space
        |> Seq.filter (fun line ->                                      // take each line
            if isCommaSeperated && line.StartsWith(";") then false          // separate by commas or semicolons
            else true
            )   
        |> (if hasHeader then Seq.skip 1 else id)                           // separate headers from data
        |> Seq.map (fun line -> line.Split(if isCommaSeperated then ',' else ';') |> Array.map (fun value -> value.Trim())) // this give you an array of elements from the comma seperated fields. We trim to make sure that any white space is removed.
        
    //// DISTANCE FUNCTIONS


    let createAttributeDetails (s:Point seq) =
        let createCategoricalPercentages (s:Point seq) =
                let head = s |> Seq.head
                if head.cls.IsNone then
                    Map.empty
                else
                    s
                    |> Seq.map (fun s -> s.categoricalAttributes |> Seq.map (fun attr -> s.cls.Value, attr))
                    |> Seq.transpose
                    |> Seq.mapi (fun i attrs ->
                        let clsAttrCntMap =
                            attrs
                            |> Seq.groupBy fst
                            |> Seq.map (fun (cls,s) ->
                                cls,
                                s
                                |> Seq.groupBy snd
                                |> Seq.map (fun (attr,s) -> attr, s |> Seq.length)
                                |> Map.ofSeq
                            )
                            |> Map.ofSeq
                        let attrCntMap =
                            clsAttrCntMap
                            |> Seq.collect (fun (KeyValue(_,attrCnts)) -> attrCnts |> Seq.map (fun (KeyValue(attr,cnt)) -> attr,cnt))
                            |> Seq.groupBy fst
                            |> Seq.map (fun (attr,cnts) -> attr, cnts |> Seq.sumBy snd)
                            |> Map.ofSeq
                        i,
                        clsAttrCntMap
                        |> Map.map (fun key attrCnt ->
                            attrCnt
                            |> Map.map (fun attr cnt ->
                                let c = attrCntMap.[attr] |> float
                                (float cnt)/c
                            )
                        )
                    )
                    |> Map.ofSeq
        let createRegressionPercentages (s:Point seq) =
                let head = s |> Seq.head
                if head.regressionValue.IsNone then
                    Map.empty
                else
                    s
                    |> Seq.map (fun s -> s.categoricalAttributes |> Seq.map (fun attr -> s.regressionValue.Value, attr))
                    |> Seq.transpose
                    |> Seq.mapi (fun i attrs ->
                        let avgRegressionValue = attrs |> Seq.averageBy fst
                        let attrRegressionAvgMap =
                            attrs
                            |> Seq.groupBy snd
                            |> Seq.map (fun (attr,s) ->
                                attr,
                                s
                                |> Seq.averageBy fst
                            )
                            |> Map.ofSeq
                        i,
                        attrRegressionAvgMap
                        |> Map.map (fun key attrRegressionAvg ->
                            attrRegressionAvg/avgRegressionValue
                        )
                    )
                    |> Map.ofSeq
        {
            categoricalAttributePercentages = createCategoricalPercentages s
            regressionAverages = createRegressionPercentages s
        }

    let regressionDistance p (point:Point) (target:Point) (attrDetails:AttributeDetails) =
        point.categoricalAttributes
        |> Seq.mapi (fun i vi ->
            let attrMap = attrDetails.regressionAverages.[i]
            let vj = target.categoricalAttributes.[i]
            let ri = match attrMap.TryFind vi with | None -> 0. | Some v -> v
            let rj = match attrMap.TryFind vj with | None -> 0. | Some v -> v
            abs((ri)-(rj))**p
        )
        |> Seq.sum
        |> fun sum -> sum**(1./p)


    let getCategoricalRegressionDistance (point:Point) (target:Point) (attributeDetails:AttributeDetails) (p:float) = 
        regressionDistance p point target attributeDetails

    let getCategoricalClassificationDistance (point:Point) (target: Point) (attributeDetails:AttributeDetails) p= 

        let sigma2 p i (vi,vj) (counts:Map<string,Map<string,float>>) =
            (counts
                |> Seq.map (fun (KeyValue(k,_)) -> k))
            |> Seq.sumBy (fun key ->
                let cia_ci =
                    match Map.tryFind key counts with
                    | Some m ->
                        match Map.tryFind vi m with
                        | None -> 0.
                        | Some v -> v
                    | None -> 0.
                let cja_cj =
                    match Map.tryFind key counts with
                    | Some m ->
                        match Map.tryFind vj m with
                        | None -> 0.
                        | Some v -> v
                    | None -> 0.
                abs((cia_ci)-(cja_cj))**p
            )

        let delta2 p (point:Point) (target:Point) (attrDetails:AttributeDetails) =
            point.categoricalAttributes
            |> Seq.mapi (fun i vi ->
                let vj = target.categoricalAttributes.[i]
                sigma2 p i (vi,vj) attrDetails.categoricalAttributePercentages.[i]
            ) 
            |> Seq.sum
            |> fun sum -> sum**(1./p)

        delta2 p point target attributeDetails
    


    type PointImpl = 
        {
            realAttributes :float[]
            categoricalAttributes: string[]
            cls : string option
            regressionValue : float option
            pValue: float
        }
        with 
            member this.distance (p:Point) (attributeDetails:AttributeDetails) = 
                if this.categoricalAttributes.Length = 0 then 
                    System.Math.Sqrt((Seq.zip this.realAttributes p.realAttributes|> Seq.sumBy (fun (a,b) -> (a-b)*(a-b)))**2.)
                else 
                    if this.cls.IsSome then 
                        System.Math.Sqrt((Seq.zip this.realAttributes p.realAttributes|> Seq.sumBy (fun (a,b) -> (a-b)*(a-b)))**2. + (getCategoricalClassificationDistance this p attributeDetails this.pValue)**2.)
                    else 
                        System.Math.Sqrt((Seq.zip this.realAttributes p.realAttributes|> Seq.sumBy (fun (a,b) -> (a-b)*(a-b)))**2. + (getCategoricalRegressionDistance this p attributeDetails this.pValue)**2. )
            interface Point with 
                member this.cls = this.cls
                member this.regressionValue = this.regressionValue
                member this.realAttributes = this.realAttributes
                member this.categoricalAttributes = this.categoricalAttributes
                member this.distance (p:Point) (attributeDetails:AttributeDetails)= this.distance p attributeDetails 
                    

    
    ////GET THE DATASET
    let fullDataset filename (classIndex:int option) (regressionIndex : int option) (pValue:float) isCommaSeperated hasHeader= 
        let classIndex,regressionIndex = 
            match classIndex,regressionIndex with 
            | None,None     -> -1,-1
            | None,Some a   -> -1,a 
            | Some a,None   -> a,-1
            | Some a,Some b -> a,b
        let dataSet = fetchTrainingSet filename isCommaSeperated hasHeader

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
            

        dataSet
        |> Seq.map (fun p -> 
            {
                cls = match classIndex with | -1 -> None | i -> Some p.[i]
                regressionValue = match regressionIndex with | -1 -> None | i -> (p.[i] |> System.Double.tryParse) //Needs to be able to parse ints into floats
                realAttributes = p |> Seq.filterWithIndex (fun i a -> realIndexes.Contains i) |>Seq.map System.Double.Parse |> Seq.toArray
                categoricalAttributes = p |> Seq.filterWithIndex (fun i a -> categoricalIndexes.Contains i) |> Seq.toArray
                pValue = pValue                
            }:>Point            
        ) |> Seq.toArray
        
    type ProcessedDataSet = {
        datasetCategoricalAttributesValues: string [] []
        datasetRealAttributeValues: float [] []
        datasetClasses: string []
    }

    let processTrainingDataset (points:Point seq) = 
        let datasetCategoricalAttributesValues= (points |> Seq.map (fun p -> p.categoricalAttributes) |> Seq.transpose |>Seq.map (fun aList -> aList |> Seq.distinct|>Seq.toArray)|>Seq.toArray) // processes the categories
        let datasetRealAttributeValues= (points|> Seq.map (fun p -> p.realAttributes) |> Seq.transpose |>Seq.map (fun aList -> aList |> Seq.distinct|>Seq.toArray)|>Seq.toArray)  // processes the attributes
        let datasetClasses= (points|>Seq.map (fun p -> p.cls)|>Seq.distinct|>Seq.choose id|>Seq.toArray) // processes the classes
        {datasetCategoricalAttributesValues=datasetCategoricalAttributesValues;datasetRealAttributeValues=datasetRealAttributeValues;datasetClasses=datasetClasses} 


        
        
    //// Loss Functions
        
    let zeroOneLoss (x:(string*string) seq) = //This implements 0-1 loss which we found was accurate enough for our analyses for classification problems
        x                                   //take the tuples..
        |>Seq.averageBy (function           //and take the average of the following matches 
            | a,b when a=b -> 0.            //if both parts of the tuple are the same, then there is an error of 0 (no error)
            | _ -> 1.                       //otherwise there is an error of 1 (lots of error)
        )                                   //effectively we are computing the percentage of the classifications that are wrong in the validation set

    let meanSquaredError (x:(float*float) seq) = //This implements MSE loss which we found was accurate enough for our analyses for regression problems
        x                                        //take the tuples
        |>Seq.averageBy (fun (a,b) -> (a-b)**2.)  //and average them all by squaring the difference of their parts
    ////


    // K-nearest Neighbor (KNN)
    // this makes this function only visible inside the defining module
    let private kNearestNeighborClassificationImpl k (trainingSet:Point seq) (attributeDetails:AttributeDetails) (p:Point) =
        //printfn "trainingSet Length %i" (trainingSet|>Seq.length)
        trainingSet
        |> Seq.sortBy (fun tp -> tp.distance p attributeDetails) // you are getting the data from the data set 
        |> Seq.take k                                       //counting all by a k
        |> Seq.map (fun tp -> tp.cls)                       //maps the function to the class it is in
        |> Seq.countBy id
        |> Seq.maxBy snd                                    // compares them and when we get the same we return that value
        |> fst                                              // compares the first and then if same returns them
    
    let private kNearestNeighborRegressionImpl k (trainingSet:Point seq) (attributeDetails:AttributeDetails) (p:Point) =
        trainingSet
        |> Seq.sortBy (fun tp -> tp.distance p attributeDetails) // you are getting the data from the data set
        |> Seq.take k                                       //counting all by a k
        |> Seq.map (fun tp -> tp.regressionValue)
        |> Seq.map (fun v -> match v with |None -> 0. | Some v -> v) // comparing each point until you get centeralized point check if each point matches with the regression point 
        |> Seq.average                                              //get the average of the points
    

    // Function to classify points via KNN
    let kNearestNeighborClassification k (trainingSet:Point seq) =
        let attributeDetails = createAttributeDetails trainingSet 
        { new Classifier with                                                               // classifier object 
            member __.classify p = kNearestNeighborClassificationImpl k trainingSet attributeDetails p
        }

    // Function to regress points via KNN
    let kNearestNeighborRegression k (trainingSet:Point seq) =
        let attributeDetails = createAttributeDetails trainingSet 
        { new Regresser with                                                                // regressor object 
            member __.regress p = kNearestNeighborRegressionImpl k trainingSet attributeDetails p
        }

    // Condensed Nearest Neighbor (CNN)
    // Implementation of Condensed Nearest Neighbor
    let createCondensedKNearestNeighborSet k (dataSet:Point seq) =
        let dataSet = ResizeArray(dataSet)                              // copy the data set array
        let results = ResizeArray()                                     // create a new array of results
        let rec loop i =                                                // iterate through the array ... 
            if i<dataSet.Count then                             // if the index falls within the data set bounds ...
                let attributeDetails = createAttributeDetails dataSet 
                let p = dataSet.[i]                                     // take the point at index i
                let computedClass =                                     // compute the sequence through nearest neighbor        
                    dataSet                                             // take the original data set
                    |> Seq.sortBy (fun tp -> tp.distance p attributeDetails)     // sort by distance
                    |> Seq.take k                                       // grab the first k nearest points
                    |> Seq.map (fun tp -> tp.cls.Value)                 // get the classes of the points
                    |> Seq.countBy id                                   // return the number of classes in those k neighbors
                    |> Seq.maxBy snd                                    // find the most common class amongst those k neighbors
                    |> fst                                              // return the majority class
                if computedClass <> p.cls.Value then                // if the computed class matches the actual class ...
                    dataSet.RemoveAt(i)                                 // remove the data point
                    loop i                                              // continue the loop
                else                                                // else ...
                    results.Add(p)                                      // add the data point to the results data set
                    loop (i+1)                                          // continue the loop
            else                                                // else
                results:>_ seq                                      // output the results to a sequence
        loop 0                                              // end the loop
        
    // Edited Nearest Neighbor (ENN)
    // Implementation of Edited Nearest Neighbor
    let createKEditedNearestNeighbor k (trainingSet:Point []) =
        trainingSet                                 // take the trainingSet
        |> Seq.mapi ( fun i point ->                // take a point and the index of said point
            (kNearestNeighborClassification k (     // we are making a kNNclassifier
                trainingSet                         // using the trainingSet                        //we are taking out the point we are looking at...
                |> Seq.mapi (fun i x -> (x,i))      // make each point into a point and it's index
                |> Seq.filter (fun (_,j) -> j<>i)   // make sure the index you are looking at is not our input point
                |> Seq.map (fun (x,_) -> x)         // take the (point,index) tuple and return just the point
                |> Seq.toArray)                     // then we make this whole ugly thing into an array again
                ).classify point                    // then we classify the point in question
            ) 
        |> Seq.mapi (fun i x -> (x,i))              // make a tuple of our point and it's index (this is so we can grab the original point's class
        |> Seq.filter (fun (x,i) ->                 // take the tuple...
            x <> trainingSet.[i].cls         // filter out points that are falsely classified
        )
        |> Seq.map (fun (_,i) -> trainingSet.[i])   // return the points from the original trainingSet

    // kMeans = (trainingSet: point seq) -> (k:int) -> ((clusterMean,Cluster):Point*Point seq) seq
   
    let kMeansImpl k (trainingSet:Point seq) =
        let temp = seq {                                //make a sequence                           //get k memes from T
            let rnd = System.Random()                   //init randomnumbergenerator
            let data = ResizeArray(trainingSet)         //convert our dataset to a resizable array
            let getRandomElement() =                    //Get a random element out of data
                if data.Count <= 0 then None            //if our data is empty return nothing
                else
                    let idx = rnd.Next(0,data.Count)    //get a random index between 0 and |data|
                    let e = data.[idx]                  //get the element e from idx
                    data.RemoveAt(idx) |> ignore        //remove the element e from data
                    Some e                              //return e
            let rec loop count =                        // this is recursive
                seq {                                   //we are making a sequence
                    if count < k then                   //if the current count is lower than k
                        match getRandomElement() with   //get a random element out of our training set and match it with...
                        | Some x ->                     //if we get some value
                            yield x                     //then return said value
                            yield! loop (count+1)       //and retun the loop
                        | _ -> ()                       //else return nothing
                }
            yield! loop 0                               //call loop with the count of 0 and return the resulting sequence
        }
        
        let firstCentroids = temp|>Seq.toArray              //we are making firstMeans an array so we can modify the means
            //gets average 
        let pointsMidpoint (ps :Point seq) =                                                        // get the mean of the points in a sequence
            let realAttributeCount = ps|>Seq.head|>(fun p -> p.realAttributes |>Array.length)       // set a value for the number of real attributes
            let realAttributes = Array.init realAttributeCount (fun i ->                            // set a value for the average value of real attributes    
                ps|>Seq.averageBy (fun x -> x.realAttributes.[i])                                       // iterate through each attribute to calculate the average
            )
            let categoricalAttributeCount = ps|>Seq.head|>(fun p -> p.categoricalAttributes |>Array.length)     // set a value for the number of categorical attributes
            let categoricalAttributes = Array.init categoricalAttributeCount (fun i ->                          // set a value for the average of the categorical attributes
                ps|>Seq.map (fun x -> x.categoricalAttributes.[i])|>Seq.countBy id |> Seq.maxBy snd |> fst          // iterate through each attribute to calculate the average  
            )
            let avgRegressionValue = ps|>Seq.averageBy (fun p -> (p.regressionValue|>(fun v -> match v with |None -> 0. |Some v -> v)))     // calculate the average regression from both the real and categorical attributes
            let maxClassValue = ps|> Seq.map (fun x -> x.cls)|>Seq.countBy id |> Seq.maxBy snd |> fst                                           // iterate through all attributes to calculate the average  

           
            // assign the class value, regression value, real and categorical attributes to the point object
            {
                cls = maxClassValue
                regressionValue = Some avgRegressionValue
                realAttributes = realAttributes
                categoricalAttributes = categoricalAttributes
                pValue = (ps|>Seq.head|>unbox<PointImpl>).pValue
            } :> Point


        // this is getting the centroids of the points 
        let getCentroids (centroids:Point seq) (dataset:Point seq) =
            let epsilon = 0.0001
            let attributeDetails = createAttributeDetails dataset
            let rec loop (centroids':Point seq) =
                dataset                                                             //iterate through our data set
                |>Seq.groupBy (fun p->centroids'|>Seq.minBy (fun c -> (p.distance c attributeDetails)))                                                  //group by the centroids
                |>Seq.map (fun (m,ps) ->                                            
                    m,(ps |> pointsMidpoint),ps
                )
                |> (fun centroids'' ->
                    match (centroids''|>Seq.sumBy(fun (oldCentroid,newCentroid,_)->oldCentroid.distance newCentroid attributeDetails)) with 
                    | v when v > epsilon -> loop (centroids'' |> Seq.map (fun (_,b,_) -> b))
                    | _ -> (centroids'' |> Seq.map (fun (_,b,c) -> b,c))
                )
            loop centroids
        
        getCentroids firstCentroids trainingSet //Get the K means...
        
    
    let kMedoids k (trainingSet:Point seq) = 
        let attributeDetails = createAttributeDetails trainingSet
        let distortion (point:Point) (targets:Point seq) = 
            targets
            |> Seq.sumBy(fun p -> point.distance p attributeDetails)
         
        kMeansImpl k trainingSet
        |> Seq.map (fun (m,ps)-> 
            seq {
                yield (m,distortion m ps)
                yield! ps|>Seq.map(fun p -> (p,distortion p ps))
            }
            |>Seq.minBy snd
            |>fst
        )
    
    let classifyOnCentroids centroids point = 
        (kNearestNeighborClassification 1 centroids).classify point
    
// FUNCTION IMPLEMENTATIONS
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
// Open project
open Project2

// Load data sets
let ds1 = (fullDataset @"D:\Fall2019\Machine Learning\Project 2\Data\abalone.data" (Some 0) None 2. true false) //filename classIndex regressionIndex pValue isCommaSeperated hasHeader
let ds2 = (fullDataset @"D:\Fall2019\Machine Learning\Project 2\Data\car.data" (Some 6) None 2. true false)
let ds3 = (fullDataset @"D:\Fall2019\Machine Learning\Project 2\Data\forestfires.csv" None (Some 12) 2. true true)
let ds4 = (fullDataset @"D:\Fall2019\Machine Learning\Project 2\Data\machine.data" None (Some 9) 2. true false )
let ds5 = (fullDataset @"D:\Fall2019\Machine Learning\Project 2\Data\segmentation.data" (Some 0) None 2. true true)
let ds6 = (fullDataset @"D:\Fall2019\Machine Learning\Project 2\Data\winequality-red.csv" None (Some 9) 2. false true)
let ds7 = (fullDataset @"D:\Fall2019\Machine Learning\Project 2\Data\winequality-white.csv" None (Some 11) 2. false true)


///////////////////////////
//// k-fold

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

type KfoldType = 
    | Classification of (Point seq -> Classifier)*((string*string) seq -> float)
    | Regression of (Point seq -> Regresser)*((float*float) seq -> float)
    

let applyKFold (trainingSet:Point seq) (validationSet: Point seq) (preprocessFunction: (Point [] -> Point seq) option) (kfoldType) =   //apply the loss function (MSE) to the kth fold
    
    let workingTrainingSet = (match preprocessFunction with |Some f -> (f (trainingSet|>Seq.toArray)) |_ -> trainingSet)//preprocess our validation set if we have a preprocessing function, otherwise use our validation set
    match kfoldType with 
    | Classification (classifier,lossFunction) -> 
        let classifier = classifier workingTrainingSet
        validationSet
        |> Seq.map (fun x -> (classifier.classify x).Value,x.cls.Value
        )                                                                                //grab each element out of it and run it as the "sample" in our classify function and pair the resultant class with the element's ACTUAL class in a tuple
        |> lossFunction                                                                 //run the loss algorithm with the sequence of class tuples
    | Regression (regresser,lossFunction) -> 
        let regresser = regresser workingTrainingSet
        validationSet
        |> Seq.map (fun x -> (regresser.regress x),x.regressionValue.Value
        )                                                                                //grab each element out of it and run it as the "sample" in our classify function and pair the resultant class with the element's ACTUAL class in a tuple
        |> lossFunction   
    //                                                                              //The result is a float: the % of elements that were guessed incorrectly

let doKFold k (dataSet:Point seq) (preprocessFunction: (Point [] -> Point seq) option) kfoldType =           //This is where we do the k-folding algorithim this will return the average from all the kfolds
    let sw = System.Diagnostics.Stopwatch.StartNew()
    try 
        let folds = getRandomFolds k dataSet    //Lets get the k folds randomly using the function above; this returns an array of Data seqences
        //printfn "Doing kFold with k:%d" k
        Seq.init k (fun k ->                    //We're initializing a new seq of size k using the lambda function "(fun k -> ...)"" to get the kth element
            let validationSet = folds.[k]       //The first part of our function we get the validation set by grabing the kth data Seq from folds
            let trainingSet =                   //The training set requires us to do a round-about filter due to the fact that seqences are never identical and we can't use a Seq.filter...
                folds                           //lets grab the array of data seqences
                |> Seq.mapi (fun i p -> (i,p))  //each seqence in the array is mapped to a tuple with the index of the sequence as "(index,sequence)"
                |> Seq.filter(fun (i,_) -> i<>k)//now we will filter out the seqence that has the index of k
                |> Seq.collect snd              //now we grab the seqence from the tuple
            applyKFold trainingSet validationSet preprocessFunction kfoldType 
        )   //Finally lets apply our function above "applyKFold" to our training set and validation set using our preproccess function, lossfunction, and algorithm
        |> Seq.average                          //the result is a seq of floats so we'll just get the average our % failuresto give us a result to our k-fold analysis as the accuracy of our algorithm
    finally 
        sw.Stop()
        printfn " ... Elapsed Time: %0.03f" sw.Elapsed.TotalSeconds

    //Currently an issue 

///////////////////////////
// Run each of the following:
//  - K-nearest Neighbor with original classification data sets, 10-Fold, 0/1 Loss (ds1)
//  - K-nearest Neighbor with original regression data sets, 10-Fold, MSE (ds2)
//  - K-nearest Neighbor with ENN data sets, 10-Fold, 0/1 Loss (ds3)
//  - K-nearest Neighbor with CNN data sets, 10-Fold, 0/1 Loss (ds4)
//  - K-nearest Neighbor with K-means data sets, 10-Fold, 0/1 Loss (ds5)
//  - K-nearest Neighbor with ___ data sets, 10-Fold, 0/1 Loss (ds6)
//  - K-nearest Neighbor with ___ data sets, 10-Fold, 0/1 Loss (ds7)

//r b b 
//r b b 
//r r r






//let sw = System.Diagnostics.Stopwatch.StartNew()
let res = 
    seq {3;6;10}
    |>Seq.map (fun k' ->
        [|ds1;ds2;ds3;ds4;ds5;ds6;ds7|]
        |> Seq.mapi (fun i ds -> printfn "ds%d" (i+1); ds)
        |> Seq.map (fun ds -> ds,(if ds.[1].cls.IsSome then printf "\tknn Class "; doKFold 10 ds None (Classification((kNearestNeighborClassification k' ),zeroOneLoss))   else 0.0))
        |> Seq.map (fun (ds,knnc) -> ds,(knnc,(if ds.[1].cls.IsNone then printf "\tknn Regress "; doKFold 10 ds None (Regression((kNearestNeighborRegression k' ), meanSquaredError))  else 0.0)))
        |> Seq.map (fun (ds,(knnc,knnr)) -> ds,(knnc,knnr,(if ds.[1].cls.IsSome then printf "\tenn-knn "; doKFold 10 ds (Some (createKEditedNearestNeighbor k')) (Classification((kNearestNeighborClassification k' ),zeroOneLoss)) else 0.0)))
        |> Seq.map (fun (ds,(knnc,knnr,enn)) -> ds,(knnc,knnr,enn,(if ds.[1].cls.IsSome then printf "\tcnn-knn "; doKFold 10 ds (Some (createCondensedKNearestNeighborSet k')) (Classification((kNearestNeighborClassification k'),zeroOneLoss)) else 0.0)))
        |> Seq.map (fun (ds,(knnc,knnr,enn,cnn)) -> ds,(knnc,knnr,enn,cnn,(if ds.[1].cls.IsSome then printf "\tkm-knn"; doKFold 10 ds (Some (fun d -> createKEditedNearestNeighbor k' d|> kMeansImpl k' |>Seq.map fst )) (Classification((kNearestNeighborClassification k'),zeroOneLoss)) else 0.0)))
        |> Seq.map (fun (ds,(knnc,knnr,enn,cnn,km)) -> ds,(knnc,knnr,enn,cnn,km,(if ds.[1].cls.IsSome then printf "\tpam-knn"; doKFold 10 ds (Some (fun d -> createKEditedNearestNeighbor k' d|> kMedoids k')) (Classification((kNearestNeighborClassification k'),zeroOneLoss)) else 0.0)))
        |> Seq.map (fun (_,(knnc,knnr,enn,cnn,km,pam)) ->(knnc,knnr,enn,cnn,km,pam))
        |>Seq.toArray
    
    )
    |>Seq.toList
//sw.Stop()
//sw.Elapsed.TotalSeconds

let trainingSet = ds2

//let ds = (fullDataset "D:\Fall2019\Machine Learning\Project 2\Data\machine.data" None (Some 9) 2.)

//let classifier1 = kNearestNeighborClassification 2 ds
//let classifier2 = KNearestNeighborClassification(2,ds)
//let regressor1 = kNearestNeighborRegression 2 ds

//let trainingSet = ds

//let p=ds.[1]

// END OF CODE


// SPREAD THESE EVERYWHERE! printfn("⣿⣿⣿⡇⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿"); printfn("⣿⣿⣿⡇⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");printfn("⣿⣿⣿⡇⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");printfn("⣿⣿⣿⡇⠄⣿⣿⣿⡿⠟⠋⣉⣉⣉⡙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");printfn("⣿⣿⣿⠃⠄⠹⠟⣡⣶⡿⢟⣛⣛⡻⢿⣦⣩⣤⣤⣤⣬⡉⢻⣿⣿⣿⣿⣿⣿⣿");printfn("⣿⣿⣿⠄⢀⢤⣾⣿⣿⣿⣿⡿⠿⠿⠿⢮⡃⣛⣛⡻⠿⢿⠈⣿⣿⣿⣿⣿⣿⣿");printfn("⣿⡟⢡⣴⣯⣿⣿⣿⣉⠤⣤⣭⣶⣶⣶⣮⣔⡈⠛⠛⠛⢓⠦⠈⢻⣿⣿⣿⣿⣿");printfn("⠏⣠⣿⣿⣿⣿⣿⣿⣿⣯⡪⢛⠿⢿⣿⣿⣿⡿⣼⣿⣿⣿⣶⣮⣄⠙⣿⣿⣿⣿");printfn("⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣾⡭⠴⣶⣶⣽⣽⣛⡿⠿⠿⠿⠿⠇⣿⣿⣿⣿");printfn("⣿⣿⣿⣿⣿⣿⣿⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣝⣛⢛⡛⢋⣥⣴⣿⣿⣿⣿⣿");printfn("⣿⣿⣿⣿⣿⢿⠱⣿⣿⣛⠾⣭⣛⡿⢿⣿⣿⣿⣿⣿⣿⣿⡀⣿⣿⣿⣿⣿⣿⣿");printfn("⠑⠽⡻⢿⣿⣮⣽⣷⣶⣯⣽⣳⠮⣽⣟⣲⠯⢭⣿⣛⣛⣿⡇⢸⣿⣿⣿⣿⣿⣿");printfn("⠄⠄⠈⠑⠊⠉⠟⣻⠿⣿⣿⣿⣿⣷⣾⣭⣿⣛⠷⠶⠶⠂⣴⣿⣿⣿⣿⣿⣿⣿");printfn("⠄⠄⠄⠄⠄⠄⠄⠄⠁⠙⠒⠙⠯⠍⠙⢉⣉⣡⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");printfn("⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿")
//printf("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⣶⣶⣦⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣷⣤⠀⠈⠙⢿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⠆⠰⠶⠀⠘⢿⣿⣿⣿⣿⣿⣆⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⠏⠀⢀⣠⣤⣤⣀⠙⣿⣿⣿⣿⣿⣷⡀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⢠⠋⢈⣉⠉⣡⣤⢰⣿⣿⣿⣿⣿⣷⡈⢿⣿⣿⣿⣿⣷⡀
//⠀⠀⠀⠀⠀⠀⠀⡴⢡⣾⣿⣿⣷⠋⠁⣿⣿⣿⣿⣿⣿⣿⠃⠀⡻⣿⣿⣿⣿⡇
//⠀⠀⠀⠀⠀⢀⠜⠁⠸⣿⣿⣿⠟⠀⠀⠘⠿⣿⣿⣿⡿⠋⠰⠖⠱⣽⠟⠋⠉⡇
//⠀⠀⠀⠀⡰⠉⠖⣀⠀⠀⢁⣀⠀⣴⣶⣦⠀⢴⡆⠀⠀⢀⣀⣀⣉⡽⠷⠶⠋⠀
//⠀⠀⠀⡰⢡⣾⣿⣿⣿⡄⠛⠋⠘⣿⣿⡿⠀⠀⣐⣲⣤⣯⠞⠉⠁⠀⠀⠀⠀⠀
//⠀⢀⠔⠁⣿⣿⣿⣿⣿⡟⠀⠀⠀⢀⣄⣀⡞⠉⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⡜⠀⠀⠻⣿⣿⠿⣻⣥⣀⡀⢠⡟⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
//⢰⠁⠀⡤⠖⠺⢶⡾⠃⠀⠈⠙⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
//⠈⠓⠾⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
