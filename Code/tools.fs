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


// SEQ MODULE
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
module Seq = 
    
    // Function to filter a value from a sequence by index
    let filterWithIndex (filter:int->'a->bool) (s:'a seq) =
        seq {
            let e = s.GetEnumerator()                                       // Get the type of s
            let mutable idx = 0                                             // Set index
            while e.MoveNext() do                                           // Iterate through e
                if filter idx e.Current then                                // Check for filter match
                    yield e.Current                                         // Get the e value
                idx <- idx+1                                                // Add to the index
        }

    // Function to choose a value from a sequence by index
    let chooseWithIndex (choose:int->'a->'b option) (s:'a seq) =
        seq {
            let e = s.GetEnumerator()                                       // Get the type of s
            let mutable idx = 0                                             // Set index
            while e.MoveNext() do                                           // Iterate through e
                match choose idx e.Current with                             // Match value with correct index
                | Some b -> yield b                                         // Grab the b that matches
                | _ -> ()
                idx <- idx+1                                                // Add to the index
        }


// EXTENSIONSMODULE
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
module Extensions =

    // Type to define a system Double datatype
    type System.Double with 
        static member tryParse s = match System.Double.TryParse (s:string) with | false,_ -> None | true,v -> Some v 

    // Type to define a system 32-bit Int datatype
    type System.Int32 with 
        static member tryParse s = match System.Int32.TryParse (s:string) with | false,_ -> None | true,v -> Some v 


//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
// END OF CODE   
