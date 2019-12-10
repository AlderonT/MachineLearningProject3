let rand = System.Random()
let createRandomNumber min max = 
    (max-min)*rand.NextDouble()+min

Seq.init 10000 (fun i -> 
    let x = createRandomNumber (System.Math.PI * -100.) (System.Math.PI * 100.)
    let y = createRandomNumber -100. 100.
    x,y,(if y<100.*System.Math.Cos((float x)/100.) then "red" else "blue")
    )
|> Seq.map (fun (x,y,s)-> sprintf "%f,%f,%s" x y s)
|> String.concat "\n"
|> fun text -> System.IO.File.WriteAllText(System.IO.Path.Combine (__SOURCE_DIRECTORY__,"testing.csv"),text)

System.Environment.CurrentDirectory