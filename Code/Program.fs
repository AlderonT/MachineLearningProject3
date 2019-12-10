module Autoencoder

    type Network = 
        {            
            weights: float32[,][]
            deltas: float32[][]
            layers: float32[][]
        }
        with
            member this.outputLayer = this.layers.[this.layers.Length-1]
            member this.inputLayer = this.layers.[0]
            member this.outputLayerDeltas = this.deltas.[this.layers.Length-1]
            member this.inputLayerDeltas = this.deltas.[0]

    let dotProduct (x:float32[]) (M:float32[,]) (r:float32[]) =
        if x.Length < M.GetLength(1) || r.Length < M.GetLength(0) then
            failwithf "Can't dot x[%d] by M[%d,%d] to make r[%d] " x.Length (M.GetLength(0)) (M.GetLength(1)) r.Length
        let width,height = M.GetLength(1), M.GetLength(0)
        for j = 0 to height-1 do
            let mutable sum = 0.f
            for i = 0 to width-1 do
                sum <- sum + x.[i]*M.[j,i]
            r.[j] <- sum

    let logistic (x:float32[]) (r:float32[]) =
        if r.Length < x.Length then
            failwithf "r[%d] is too short for x[%d]" r.Length x.Length
        for i = 0 to x.Length-1 do
            let x' = x.[i]
            let v = 1. / 1. - (System.Math.Exp(-(float x')))
            r.[i] <- float32 v

    let outputDeltas (outputs:float32[]) (expected:float32[]) (deltas:float32[]) =
        for i = 0 to expected.Length-1 do
            let o = outputs.[i]
            let t = expected.[i]
            deltas.[i] <- (o-t)*o*(1.f-o)       //(output - target)*output*(1-output)
    let innerDeltas (weights:float32[,]) (inputs:float32[]) (outputDeltas:float32[]) (deltas:float32[]) =
        for j = 0 to inputs.Length-1 do
            let mutable sum = 0.f
            for l = 0 to outputDeltas.Length-1 do
                let weight = weights.[l,j]
                sum <- outputDeltas.[l]*weight + sum
            deltas.[j] <- sum*inputs.[j]*(1.f-inputs.[j])
           
    let updateWeights learningRate (weights:float32[,]) (inputs:float32[]) (outputDeltas:float32[]) =
        for j = 0 to outputDeltas.Length-1 do
            for i = 0 to inputs.Length-1 do
                let weight = weights.[j,i]
                let delta = -learningRate*inputs.[i]*outputDeltas.[j]
                weights.[j,i] <- weight + delta

    let backprop learningRate (network: Network) (expectedOutputs:float32[]) =
        let outputLayer = network.outputLayer
        outputDeltas outputLayer expectedOutputs outputLayer.deltas
        for j = network.connections.Length-1 downto 1 do    
            let connectionMatrix = network.connections.[j]
            let inLayer = connectionMatrix.inputLayer
            let outlayer = connectionMatrix.outputLayer
            innerDeltas connectionMatrix.weights inLayer.nodes outlayer.deltas inLayer.deltas
            updateWeights learningRate connectionMatrix.weights inLayer.nodes outlayer.deltas
        let connectionMatrix = network.connections.[0]
        updateWeights learningRate connectionMatrix.weights connectionMatrix.inputLayer.nodes connectionMatrix.outputLayer.deltas
        

[<EntryPoint>]
let main argv =
    //let dsmd1 = (fullDataset @"..\Data\abalone.data" (Some 0) None 2. true false) //filename classIndex regressionIndex pValue isCommaSeperated hasHeader
    //let dsmd2 = (fullDataset @"..\Data\car.data" (Some 6) None 2. true false)
    //let dsmd3 = (fullDataset @"..\Data\forestfires.csv" None (Some 12) 2. true true)
    //let dsmd4 = (fullDataset @"..\Data\machine.data" None (Some 9) 2. true false )
    //let dsmd5 = (fullDataset @"..\Data\segmentation.data" (Some 0) None 2. true true)
    //let dsmd6 = (fullDataset @"..\Data\winequality-red.csv" None (Some 9) 2. false true)
    //let dsmd7 = (fullDataset @"..\Data\winequality-white.csv" None (Some 11) 2. false true)
    //let datasets = [|dsmd1;dsmd2;dsmd3;dsmd4;dsmd5;dsmd6;dsmd7|]
    ////let ds1,metadata = (fullDataset @"D:\Fall2019\Machine Learning\MachineLearningProject3\Data\car.data" (Some 6) None 2. true false) //filename classIndex regressionIndex pValue isCommaSeperated hasHeader
    
    datasets |> ignore
    0