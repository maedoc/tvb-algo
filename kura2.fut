let pi_2 = 6.28318530718f32

let wrap_2_pi x =
  if x < 0f32 then x + pi_2 else
  if x > pi_2 then x - pi_2 else x

-- ring here is state(:, j_node)
-- nh should be power of 2 to allow for a shift instead of mod
-- should premultiply the lengths
let aff_ij (t:i32) (rj:[nh]f32) (wij:f32) (dij:i32) (theta_i:f32) = 
  if wij == 0f32 then 0f32 else
  let theta_j = rj[(t - dij + nh) % nh]
  in wij * f32.sin (theta_j - theta_i)

type conn = (r:[nn][nh]f32) (w:[nn][nn]f32) (d:[nn][nn]i32)
type pars = {omega:f32, k_n:f32}
-- coupling function in TVB

let aff (t:i32) ({r,w,d}:conn) (th:[nn]f32): [nn]f32 = 
  let aff_col rj wi di = map3 (aff_ij t rj) wi di th in
  map3 aff_col rj w d |> transpose |> map (reduce (+) 0)

let step (t:i32) (dt:f32) (c:conn) (p:pars) (th:[nn]f32): [nn]f32 =
  let euler x a = x + dt * (p.omega + p.k * a)
  in zip2 th (aff c) |> map2 euler

-- util
let lengths_to_dij (lengths:[nn][nn]f32) (rec_speed_dt:f32): [nn][nn]f32 =
  map (\row -> map (\el -> el * rec_speed_dt) row) lengths

entry main
  (weights:[nn][nn]f32) 
  (lengths:[nn][nn]f32)
  (rec_speed_dt:f32)
  : i32 =
  let dij = lengths_to_dij lengths rec_speed_dt
  in 5
