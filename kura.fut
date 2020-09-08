let cf (theta_i:f32) (theta:[]f32): f32 =
    let aff = reduce (+) 0f32 (map (\theta_j -> f32.sin (theta_j - theta_i)) theta)
    in aff / (f32.i32 (length theta))

let kura (theta:[]f32) (omega:f32) (k:f32): []f32 = 
    map (\theta_i -> theta_i + 0.1f32 * (omega + k * (cf theta_i theta))) theta
    
let init (x:i32): f32 = x |> f32.i32 |> (\x -> x/1024f32) |> f32.sin

let integrate1 (nn:i32) (nt:i32) (k:f32): f32 =
  let x_ = map init (1..<nn)
  let x_ = loop x_ for t < nt do kura x_ 1f32 k
  in reduce (+) 0.0f32 x_    

entry main (nn:i32) (nt:i32) (nk:i32): []f32 =
  let ks = map init (1..<nk) in
  map (\k -> integrate1 nn nt k) ks
