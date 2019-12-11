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


// TYPES MODULE
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
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


//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
// END OF CODE   
