module cfun = {
    module type t = {
        type t
        type p
        val pre : p -> t -> t -> t
        val post : p -> t -> t
    }

    module diff = {
        type p = {a: f32}
        type t = f32
        let pre (_:p) (xj:t) (xi:t) = xj - xi
        let post (p:p) (gx:t) = p.a * gx
    }

    module km = {
        type p = {G:f32}
        type t = f32
        let pre (_:p) (xj:t) (xi:t): t = f32.sin (xj - xi)
        let post (p:p) (gx:t): t = p.G * gx
    }

    module linear = {
        type p = {a:f32}
        type t = f32
        let pre (_:p) (xj:t) (_:t): t = xj
        let post (p:p) (gx:t): t = p.a * gx
    }
}

module tracts = {
    type t [n] = {w:[n][n]f32, d:[n][n]u32}
    let apply_w [n] f (t:t [n]) = {w=map (map f) t.w, d=t.d}
    let max_w [n] (t:t [n]) = 
        let max2 x y = if x > y then x else y
        let maxn xs = reduce max2 0 xs
        let rw = map maxn t.w
        in reduce max2 0 rw
    let unit_w [n] (t:t [n]) = 
        let mw = max_w t
        let div = (/mw)
        in apply_w div t
      let ones (n:i32): t[n] =
        let w = (replicate n) (replicate n 1f32)
        let d = (replicate n) (replicate n 1u32)
        in {w=w, d=d}
}

module history = {
    module type t = {
        type idx
        type el
        type buf[nn][nh]
        val get_i [nn] [nh]: buf [nn] [nh] -> idx -> idx -> el
        val get_t [nn] [nh]: buf [nn] [nh] -> idx -> el -> el
    }

    module ring = {
        type idx = i32
        type el = f32
        type buf [nn][nh] = {r:[nn][nh]el, dt:el}
        let get_i [nn][nh] (b:buf[nn][nh]) i j = b.r[i,j]
          let get_t [nn][nh] (b:buf[nn][nh]) i t =
            let j0 = i32.f32 (t / b.dt) + nh
            let v0 = b.r[i, j0 % nh]
            let v1 = b.r[i, (j0 + 1) % nh]
            let dt = t - ((f32.i32 j0) * b.dt)
            let dx = v1 - v0
            in (dx/b.dt) * dt
          let zeros nn nh (dt:el): buf[nn][nh] =
            {dt=dt, r=replicate nn (replicate nh 0f32)}
    }

    module ring_o = ring : t -- typecheck
}

module wm (H:history.t) (C:cfun.t) = {
    --let f [nn][nh] (h:H.buf [nn][nh]) (c:c.p): H.el = H.dt
}

module neural_mass = {
  module type t = {
    type t -- state vector
    type p -- parameters
    val f : p -> t -> t -- drift
    val g : p -> t -> t -- diffusion
  }
  module excitator = {
    type t = {x:f32, y:f32}
    type p = {a:f32, tau:f32, sig:f32}
    let f (p:p) ({x,y}:t) : t =
      {x = p.tau * (x - x*x*x/3 - y),
       y = (1/p.tau) * (x - p.a)}
    let g (p:p) (_:t) : t = {x=p.sig, y=p.sig}
  }
}
-- module network (W:wm.t) (N:neural_mass.t) = {}
-- module desys = {}
-- module deint (S:desys.t) = {}

