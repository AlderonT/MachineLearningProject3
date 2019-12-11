namespace SAE_NN
module SIMD =

    open System
    open System.Runtime.Intrinsics
    open System.Runtime.Intrinsics.X86

    let inline offset (addr:nativeptr<'a>) (offset:int64) : nativeptr<'a> = ((NativeInterop.NativePtr.toNativeInt addr)+((nativeint offset)*(nativeint sizeof<'a>))) |> NativeInterop.NativePtr.ofNativeInt
    let inline lessThan (addr:nativeptr<'a>) (limit:nativeptr<'a>) = (NativeInterop.NativePtr.toNativeInt addr) < (NativeInterop.NativePtr.toNativeInt limit)
    let inline incr (addr:nativeptr<'a>) : nativeptr<'a> = ((NativeInterop.NativePtr.toNativeInt addr)+(nativeint sizeof<'a>)) |> NativeInterop.NativePtr.ofNativeInt
    let inline addByteOffset (addr:nativeptr<'a>) (byteOffset:int64) : nativeptr<'a> = ((NativeInterop.NativePtr.toNativeInt addr)+(nativeint byteOffset)) |> NativeInterop.NativePtr.ofNativeInt
    let inline offsetToVector128_float32 (addr:nativeptr<float32>) (offset:int64) : Vector128<float32> = ((NativeInterop.NativePtr.toNativeInt addr)+((nativeint offset)*(nativeint sizeof<float32>))) |> NativeInterop.NativePtr.ofNativeInt<float32> |> Avx2.LoadVector128
    let inline offsetToVector256_float32 (addr:nativeptr<float32>) (offset:int64) : Vector256<float32> = ((NativeInterop.NativePtr.toNativeInt addr)+((nativeint offset)*(nativeint sizeof<float32>))) |> NativeInterop.NativePtr.ofNativeInt<float32> |> Avx2.LoadVector256

    type IntrinsicsSupport = {
        avx2 : bool
        aes : bool
        bmi1 : bool
        bmi2 : bool
        fma : bool
        lzcnt : bool
        pclmulqdq : bool
        popcnt: bool
    }
        with
            static member Support = {
                avx2 = Avx2.X64.IsSupported
                aes = Aes.X64.IsSupported
                bmi1 = Bmi1.X64.IsSupported
                bmi2 = Bmi2.X64.IsSupported
                fma = Fma.X64.IsSupported
                lzcnt = Lzcnt.X64.IsSupported
                pclmulqdq = Pclmulqdq.X64.IsSupported
                popcnt = Popcnt.X64.IsSupported
            }

    let inline dotProductRow length (a:nativeptr<float32>) (b:nativeptr<float32>) (r:nativeptr<float32>) =
        let limit1 = length - length%64L
        let limit2 = length - length%32L
        let limit3 = length - length%4L
        let mutable i = 0L
        let mutable vacc = Vector256.Zero
        while i < limit1 do
            // -----
            let pa = offset a i
            let pb = offset b i
            let va0 = Avx2.LoadVector256(pa)                                    // 1, 0.25
            let va1 = Avx2.LoadVector256(offset pa 8L)                           // 1, 0.25
            let va2 = Avx2.LoadVector256(offset pa 16L)                           // 1, 0.25
            let va3 = Avx2.LoadVector256(offset pa 24L)                          // 1, 0.25
            let vr0 = Avx2.Multiply(va0,Avx2.LoadVector256(pb))                 // 4, 0.5
            let vr1 = Avx2.Multiply(va1,Avx2.LoadVector256(offset pb 8L))       // 4, 0.5
            let vr2 = Fma.MultiplyAdd(va2,Avx2.LoadVector256(offset pb 16L),vr0) // 4, 0.5
            let vr3 = Fma.MultiplyAdd(va3,Avx2.LoadVector256(offset pb 24L),vr1)// 4, 0.5 
            let va4 = Avx2.LoadVector256(offset pa 32L)                          // 1, 0.25
            let va5 = Avx2.LoadVector256(offset pa 40L)                          // 1, 0.25
            let va6 = Avx2.LoadVector256(offset pa 48L)                          // 1, 0.25
            let va7 = Avx2.LoadVector256(offset pa 56L)                          // 1, 0.25
            let vr4 = Fma.MultiplyAdd(va4,Avx2.LoadVector256(offset pb 32L),vr2)// 4, 0.5
            let vr5 = Fma.MultiplyAdd(va5,Avx2.LoadVector256(offset pb 40L),vr3)// 4, 0.5 
            let vr6 = Fma.MultiplyAdd(va6,Avx2.LoadVector256(offset pb 48L),vr4)// 4, 0.5
            let vr7 = Fma.MultiplyAdd(va7,Avx2.LoadVector256(offset pb 56L),vr5)// 4, 0.5 
            let vr' = Avx2.Add(vr6,vr7)                                         // 4, 0.5
            vacc <- Avx2.Add(vr',vacc)                                          // 4, 0.5

            i<-i+64L
        let mutable vacc =
            if limit1 > 0L then
                let vhigh = Avx2.ExtractVector128(vacc,1uy)
                let vlow = Avx2.ExtractVector128(vacc,0uy)
                Fma.Add(vhigh,vlow)
            else
                Vector128.Zero
        while i < limit2 do
            // -----
            let pa = offset a i
            let pb = offset b i
            let va0 = Avx2.LoadVector128(pa)                                    // 1, 0.25
            let va1 = Avx2.LoadVector128(offset pa 4L)                           // 1, 0.25
            let va2 = Avx2.LoadVector128(offset pa 8L)                           // 1, 0.25
            let va3 = Avx2.LoadVector128(offset pa 12L)                          // 1, 0.25
            let vr0 = Avx2.Multiply(va0,Avx2.LoadVector128(pb))                 // 4, 0.5
            let vr1 = Avx2.Multiply(va1,Avx2.LoadVector128(offset pb 4L))       // 4, 0.5
            let vr2 = Fma.MultiplyAdd(va2,Avx2.LoadVector128(offset pb 8L),vr0) // 4, 0.5
            let vr3 = Fma.MultiplyAdd(va3,Avx2.LoadVector128(offset pb 12L),vr1)// 4, 0.5 
            let va4 = Avx2.LoadVector128(offset pa 16L)                          // 1, 0.25
            let va5 = Avx2.LoadVector128(offset pa 20L)                          // 1, 0.25
            let va6 = Avx2.LoadVector128(offset pa 24L)                          // 1, 0.25
            let va7 = Avx2.LoadVector128(offset pa 28L)                          // 1, 0.25
            let vr4 = Fma.MultiplyAdd(va4,Avx2.LoadVector128(offset pb 16L),vr2)// 4, 0.5
            let vr5 = Fma.MultiplyAdd(va5,Avx2.LoadVector128(offset pb 20L),vr3)// 4, 0.5 
            let vr6 = Fma.MultiplyAdd(va6,Avx2.LoadVector128(offset pb 24L),vr4)// 4, 0.5
            let vr7 = Fma.MultiplyAdd(va7,Avx2.LoadVector128(offset pb 28L),vr5)// 4, 0.5 
            let vr' = Avx2.Add(vr6,vr7)                                         // 4, 0.5
            vacc <- Avx2.Add(vr',vacc)                                          // 4, 0.5

            i<-i+32L
        while i < limit3 do
            let va = Avx2.LoadVector128(offset a i)                             // 1, 0.25
            vacc <- Fma.MultiplyAdd(va,Avx2.LoadVector128(offset b i),vacc)     // 4, 0.5
            i<-i+4L
        while i < length do
            let va = Avx2.LoadScalarVector128(offset a i)
            let vb = Avx2.LoadScalarVector128(offset b i)
            vacc <- Avx2.Add(Avx2.Multiply(va,vb),vacc)
            //vacc <- Fma.MultiplyAdd(va,vb,vacc) // no point in putting in the FMUL instruction because the JIT still needs to move the result back into vacc so issues a vmovups instruction
                                                  // on the other hand using an explicit multiply then add will result in the exact two instructions since the 3 operand Add can put vacc in the output
                                                  // FMUL doesn't always work unless you doing it in a stream like above, then it works great, just make sure that the last FMUL in a loop doesn't have to
                                                  // output to an accumulator register/variable
            i<-i+1L
        let h = Avx2.HorizontalAdd(vacc,vacc)
        let h' = Avx2.HorizontalAdd(h,h)
        Avx2.StoreScalar(r,h')

    let dotProduct (a:float32[]) (m:float32[,]) (r:float32[]) =
        if a.Length <> m.GetLength(1) && r.Length <> m.GetLength(0) then
            failwithf "m[%d,%d] does not match r[%d], a[%d]" (m.GetLength(0)) (m.GetLength(1)) r.Length a.Length
        use pa = fixed a
        use pr = fixed r
        let mutable pr = pr
        let mutable pm : nativeptr<float32> = System.Runtime.CompilerServices.Unsafe.AsPointer &m.[0,0] |> NativeInterop.NativePtr.ofVoidPtr
        let aLength = a.LongLength
        let prLimit = offset pr r.LongLength
        let aByteStride = a.LongLength * (int64 sizeof<float32>)
        while lessThan pr prLimit do
            dotProductRow aLength pa pm pr
            pm <- addByteOffset pm aByteStride
            pr <- incr pr
