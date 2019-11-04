


module  Network =
    type Node = 
        abstract member layer: int
        abstract member value: float
        abstract member inputConnections: Node []
        abstract member weights : float []
        abstract member recomputeValue: unit -> Node  

    let n1 = 
    {
        layer = 0
        value = 0.1
        inputConnections = []
        weights = []
        recomputeValue = {
            layer = 0
            value = 0.1
            inputConnections = []
            weights = []
            recomputeValue = recomputeValue
        }
    }     