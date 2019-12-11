namespace SAE_NN
module Seq = 
    let filterWithIndex (filter:int->'a->bool) (s:'a seq) =
        seq {
            let e = s.GetEnumerator()
            let mutable idx = 0
            while e.MoveNext() do
                if filter idx e.Current then
                    yield e.Current
                idx <- idx+1
        }
    let chooseWithIndex (choose:int->'a->'b option) (s:'a seq) =
        seq {
            let e = s.GetEnumerator()
            let mutable idx = 0
            while e.MoveNext() do
                match choose idx e.Current with 
                | Some b -> yield b 
                | _ -> ()
                idx <- idx+1
        }
module Extensions =
    type System.Double with 
        static member tryParse s = match System.Double.TryParse (s:string) with | false,_ -> None | true,v -> Some v 
    type System.Int32 with 
        static member tryParse s = match System.Int32.TryParse (s:string) with | false,_ -> None | true,v -> Some v 