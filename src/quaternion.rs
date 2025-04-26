use std::ops;

use approx::ulps_eq;

use crate::vec::Vec3;

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
pub struct Quaternion(pub [f32; 4]);

// ------------------------------------------ Constructors ------------------------------------------
impl Quaternion {
    pub fn new(w: f32, xi: f32, yj: f32, zk: f32) -> Self {
        Self([xi, yj, zk, w])
    }
    pub fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }
    pub fn from_axis_angle(axis: Vec3, angle_rad: f32) -> Self {
        let half_angle = angle_rad / 2.0;
        let s = half_angle.sin();
        let c = half_angle.cos();
        Self::new(c, s * axis.x(), s * axis.y(), s * axis.z())
    }
    // TODO:
    // pub fn from_arc(
    //     src: Vec3,
    //     dst: Vec3,
    //     fallback: Option<Vec3>, //? What is this?
    // ) -> Self {
    //     let mag_avg = (src.length_squared() * dst.length_squared()).sqrt();
    //     let dot = src.dot(dst);
    //     if ulps_eq!(dot, &mag_avg) {
    //         Self::identity()
    //     } else if ulps_eq!(dot, -mag_avg) {
    //         let axis = fallback.unwrap_or_else(|| {
    //             let mut v = Vec3::unit_x().cross(src);
    //             if ulps_eq!(v, 0.0) {
    //                 v = Vec3::unit_y().cross(src);
    //             }
    //             v.normalize()
    //         });
    //         Quaternion::from_axis_angle(axis, Rad::turn_div_2())
    //     } else {
    //         Quaternion::from_sv(mag_avg + dot, src.cross(dst)).normalize()
    //     }
    // }
    // pub fn look_at(forward: Vec3, up: Vec3) -> Self {
    //     let right = forward.cross(up).normalized();
    //     // Imagine a 3x3 matrix where the columns are right, up, forward
    //     let mut q = Self::identity();

    //     let diagonal = right.x() + up.y() + forward.z();

    //     q
    // }
}

// ------------------------------------------ Member functions ------------------------------------------
impl Quaternion {
    pub fn w(self) -> f32 {
        self.0[3]
    }
    pub fn x(self) -> f32 {
        self.0[0]
    }
    pub fn y(self) -> f32 {
        self.0[1]
    }
    pub fn z(self) -> f32 {
        self.0[2]
    }
    pub fn mw(&mut self) -> &mut f32 {
        &mut self.0[3]
    }
    pub fn mx(&mut self) -> &mut f32 {
        &mut self.0[0]
    }
    pub fn my(&mut self) -> &mut f32 {
        &mut self.0[1]
    }
    pub fn mz(&mut self) -> &mut f32 {
        &mut self.0[2]
    }
    pub fn rotate_x(&mut self, rad: f32) {
        let half = rad / 2.0;
        *self *= Self::new(half.cos(), half.sin(), 0.0, 0.0);
    }
    pub fn rotate_y(&mut self, rad: f32) {
        let half = rad / 2.0;
        *self *= Self::new(half.cos(), 0.0, half.sin(), 0.0);
    }
    pub fn rotate_z(&mut self, rad: f32) {
        let half = rad / 2.0;
        *self *= Self::new(half.cos(), 0.0, 0.0, half.sin());
    }
}

// Macro to automatically declare operator overloads for all value and borrow type
// combinations, using the same code block as the body.
macro_rules! impl_binary_op {
    ($op:tt : $method:ident => (
           $lhs_i:ident : $lhs_t:path,
           $rhs_i:ident : $rhs_t:path
        ) -> $return_t:path $body:block
    ) => {
        impl ops::$op<$rhs_t> for $lhs_t {
            type Output = $return_t;
            fn $method(self, $rhs_i: $rhs_t) -> $return_t {
                let $lhs_i = self;
                $body
            }
        }
        impl ops::$op<&$rhs_t> for $lhs_t {
            type Output = $return_t;
            fn $method(self, $rhs_i: &$rhs_t) -> $return_t {
                let $lhs_i = self;
                $body
            }
        }
        impl ops::$op<$rhs_t> for &$lhs_t {
            type Output = $return_t;
            fn $method(self, $rhs_i: $rhs_t) -> $return_t {
                let $lhs_i = self;
                $body
            }
        }
        impl ops::$op<&$rhs_t> for &$lhs_t {
            type Output = $return_t;
            fn $method(self, $rhs_i: &$rhs_t) -> $return_t {
                let $lhs_i = self;
                $body
            }
        }
    };
}

// TODO:
impl_binary_op!(Add : add => (lhs: Quaternion, rhs: Quaternion) -> Quaternion {
    Quaternion([
        lhs.w() + rhs.w(),
        lhs.x() + rhs.x(),
        lhs.y() + rhs.y(),
        lhs.z() + rhs.z(),
    ])
});

impl_binary_op!(Sub : sub => (lhs: Quaternion, rhs: Quaternion) -> Quaternion {
    Quaternion([
        lhs.w() - rhs.w(),
        lhs.x() - rhs.x(),
        lhs.y() - rhs.y(),
        lhs.z() - rhs.z(),
    ])
});

impl_binary_op!(Mul : mul => (lhs: Quaternion, rhs: Quaternion) -> Quaternion {
    Quaternion([
            lhs.w() * rhs.w() - lhs.x() * rhs.x() - lhs.y() * rhs.y() - lhs.z() * rhs.z(),
            lhs.w() * rhs.x() + lhs.x() * rhs.w() + lhs.y() * rhs.z() - lhs.z() * rhs.y(),
            lhs.w() * rhs.y() + lhs.y() * rhs.w() + lhs.z() * rhs.x() - lhs.x() * rhs.z(),
            lhs.w() * rhs.z() + lhs.z() * rhs.w() + lhs.x() * rhs.y() - lhs.y() * rhs.x(),
    ])
});

impl_binary_op!(Mul : mul => (lhs: Quaternion, rhs: f32) -> Quaternion {
    Quaternion([
        lhs.w() * rhs,
        lhs.x() * rhs,
        lhs.y() * rhs,
        lhs.z() * rhs,
    ])
});

impl_binary_op!(Mul : mul => (lhs: f32, rhs: Quaternion) -> Quaternion {
    Quaternion([
        rhs.w() * lhs,
        rhs.x() * lhs,
        rhs.y() * lhs,
        rhs.z() * lhs,
    ])
});

impl_binary_op!(Div : div => (lhs: Quaternion, rhs: f32) -> Quaternion {
    Quaternion([
        lhs.w() / rhs,
        lhs.x() / rhs,
        lhs.y() / rhs,
        lhs.z() / rhs,
    ])
});

impl ops::AddAssign for Quaternion {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl ops::SubAssign for Quaternion {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ops::MulAssign<Quaternion> for Quaternion {
    fn mul_assign(&mut self, rhs: Quaternion) {
        *self = *self * rhs;
    }
}

impl ops::MulAssign<f32> for Quaternion {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl ops::DivAssign<f32> for Quaternion {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl ops::Neg for Quaternion {
    type Output = Quaternion;
    fn neg(self) -> Self::Output {
        Quaternion([-self.w(), -self.x(), -self.y(), -self.z()])
    }
}

impl Into<cgmath::Quaternion<f32>> for Quaternion {
    fn into(self) -> cgmath::Quaternion<f32> {
        cgmath::Quaternion::<f32> {
            s: self.w(),
            v: cgmath::Vector3 {
                x: self.x(),
                y: self.y(),
                z: self.z(),
            },
        }
    }
}

impl From<cgmath::Quaternion<f32>> for Quaternion {
    fn from(q: cgmath::Quaternion<f32>) -> Self {
        Self([q.s, q.v.x, q.v.y, q.v.z])
    }
}

impl approx::AbsDiffEq for Quaternion {
    //? Not sure why we need to add `as ...`
    type Epsilon = <f32 as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f32::abs_diff_eq(&self.w(), &other.w(), epsilon)
            && f32::abs_diff_eq(&self.x(), &other.x(), epsilon)
            && f32::abs_diff_eq(&self.y(), &other.y(), epsilon)
            && f32::abs_diff_eq(&self.z(), &other.z(), epsilon)
    }
}

impl approx::RelativeEq for Quaternion {
    #[inline]
    fn default_max_relative() -> <f32 as approx::AbsDiffEq>::Epsilon {
        f32::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: <f32 as approx::AbsDiffEq>::Epsilon,
        max_relative: <f32 as approx::AbsDiffEq>::Epsilon,
    ) -> bool {
        f32::relative_eq(&self.w(), &other.w(), epsilon, max_relative)
            && f32::relative_eq(&self.x(), &other.x(), epsilon, max_relative)
            && f32::relative_eq(&self.y(), &other.y(), epsilon, max_relative)
            && f32::relative_eq(&self.z(), &other.z(), epsilon, max_relative)
    }
}

impl approx::UlpsEq for Quaternion {
    #[inline]
    fn default_max_ulps() -> u32 {
        f32::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(
        &self,
        other: &Self,
        epsilon: <f32 as approx::AbsDiffEq>::Epsilon,
        max_ulps: u32,
    ) -> bool {
        f32::ulps_eq(&self.w(), &other.w(), epsilon, max_ulps)
            && f32::ulps_eq(&self.x(), &other.x(), epsilon, max_ulps)
            && f32::ulps_eq(&self.y(), &other.y(), epsilon, max_ulps)
            && f32::ulps_eq(&self.z(), &other.z(), epsilon, max_ulps)
    }
}

impl std::fmt::Display for Quaternion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {}i, {}j, {}k)",
            self.w(),
            self.x(),
            self.y(),
            self.z()
        )
    }
}
