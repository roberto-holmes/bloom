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
    pub fn from_euler(pitch: f32, roll: f32, yaw: f32) -> Self {
        // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        let cr = f32::cos(pitch * 0.5);
        let sr = f32::sin(pitch * 0.5);
        let cp = f32::cos(yaw * 0.5);
        let sp = f32::sin(yaw * 0.5);
        let cy = f32::cos(roll * 0.5);
        let sy = f32::sin(roll * 0.5);

        Self::new(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
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
    fn conjugate(&self) -> Self {
        Self::new(self.w(), -self.x(), -self.y(), -self.z())
    }
    fn vec(&self) -> Vec3 {
        Vec3::new(self.x(), self.y(), self.z())
        // let v = Vec3::new(self.x(), self.y(), self.z());
        // if ulps_eq!(v, Vec3::zero()) {
        //     Vec3::zero()
        // } else {
        //     v.normalized()
        // }
    }
    fn magnitude2(&self) -> f32 {
        self.w() * self.w() + self.x() * self.x() + self.y() * self.y() + self.z() * self.z()
    }
    fn magnitude(&self) -> f32 {
        self.magnitude2().sqrt()
    }
    pub fn normalise(&mut self) {
        let magnitude = self.magnitude();
        for i in &mut self.0 {
            *i /= magnitude
        }
    }
    pub fn apply(&self, v: Vec3) -> Vec3 {
        // https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
        let quat_v = self.vec();
        2.0 * quat_v.dot(v) * quat_v
            + (self.w() * self.w() - quat_v.dot(quat_v)) * v
            + 2.0 * self.w() * quat_v.cross(v)
        // ((self * Quaternion::from(v)) * self.conjugate()).vec()
    }
    pub fn rotate_x(&mut self, rad: f32) {
        let half = rad / 2.0;
        *self *= Self::new(half.cos(), half.sin(), 0.0, 0.0);
        self.normalise();
    }
    pub fn rotate_y(&mut self, rad: f32) {
        let half = rad / 2.0;
        *self *= Self::new(half.cos(), 0.0, half.sin(), 0.0);
        self.normalise();
    }
    pub fn rotate_z(&mut self, rad: f32) {
        let half = rad / 2.0;
        *self *= Self::new(half.cos(), 0.0, 0.0, half.sin());
        self.normalise();
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

impl_binary_op!(Add : add => (lhs: Quaternion, rhs: Quaternion) -> Quaternion {
    Quaternion::new(
        lhs.w() + rhs.w(),
        lhs.x() + rhs.x(),
        lhs.y() + rhs.y(),
        lhs.z() + rhs.z(),
    )
});

impl_binary_op!(Sub : sub => (lhs: Quaternion, rhs: Quaternion) -> Quaternion {
    Quaternion::new(
        lhs.w() - rhs.w(),
        lhs.x() - rhs.x(),
        lhs.y() - rhs.y(),
        lhs.z() - rhs.z(),
    )
});

impl_binary_op!(Mul : mul => (lhs: Quaternion, rhs: Quaternion) -> Quaternion {
    Quaternion::new(
            lhs.w() * rhs.w() - lhs.x() * rhs.x() - lhs.y() * rhs.y() - lhs.z() * rhs.z(),
            lhs.w() * rhs.x() + lhs.x() * rhs.w() + lhs.y() * rhs.z() - lhs.z() * rhs.y(),
            lhs.w() * rhs.y() + lhs.y() * rhs.w() + lhs.z() * rhs.x() - lhs.x() * rhs.z(),
            lhs.w() * rhs.z() + lhs.z() * rhs.w() + lhs.x() * rhs.y() - lhs.y() * rhs.x(),
    )
});

impl_binary_op!(Mul : mul => (lhs: Quaternion, rhs: f32) -> Quaternion {
    Quaternion::new(
        lhs.w() * rhs,
        lhs.x() * rhs,
        lhs.y() * rhs,
        lhs.z() * rhs,
    )
});

impl_binary_op!(Mul : mul => (lhs: f32, rhs: Quaternion) -> Quaternion {
    Quaternion::new(
        rhs.w() * lhs,
        rhs.x() * lhs,
        rhs.y() * lhs,
        rhs.z() * lhs,
    )
});

impl_binary_op!(Div : div => (lhs: Quaternion, rhs: f32) -> Quaternion {
    Quaternion::new(
        lhs.w() / rhs,
        lhs.x() / rhs,
        lhs.y() / rhs,
        lhs.z() / rhs,
    )
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
        Quaternion::new(-self.w(), -self.x(), -self.y(), -self.z())
    }
}

impl Into<[f32; 4]> for Quaternion {
    fn into(self) -> [f32; 4] {
        [self.x(), self.y(), self.z(), self.w()]
    }
}

impl Into<(f32, f32, f32, f32)> for Quaternion {
    fn into(self) -> (f32, f32, f32, f32) {
        (self.x(), self.y(), self.z(), self.w())
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
        Self::new(q.s, q.v.x, q.v.y, q.v.z)
    }
}

impl From<Vec3> for Quaternion {
    fn from(v: Vec3) -> Self {
        Self::new(0.0, v.x(), v.y(), v.z())
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
            "({:.3}, {:.3}i, {:.3}j, {:.3}k)",
            self.w(),
            self.x(),
            self.y(),
            self.z()
        )
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use num::Float;

    use super::*;

    #[test]
    fn create_x_90() {
        let q = Quaternion::from_axis_angle(Vec3::unit_x(), std::f32::consts::FRAC_PI_2);

        assert_ulps_eq!(
            q,
            Quaternion::new(2.0.sqrt() / 2.0, 2.0.sqrt() / 2.0, 0.0, 0.0)
        );
    }
    #[test]
    fn rotate_x_90() {
        let mut q = Quaternion::identity();
        // Rotate by 90 degrees
        q.rotate_x(std::f32::consts::FRAC_PI_2);

        assert_ulps_eq!(
            q,
            Quaternion::new(2.0.sqrt() / 2.0, 2.0.sqrt() / 2.0, 0.0, 0.0)
        );
    }
    #[test]
    fn apply_x_to_z() {
        let input = Vec3::unit_x();
        let q = Quaternion::from_euler(0.0, 0.0, -std::f32::consts::FRAC_PI_2);

        println!("{q}");

        assert_ulps_eq!(q.apply(input), Vec3::unit_z());
    }
    #[test]
    fn apply_x_to_y() {
        let input = Vec3::unit_x();
        let q = Quaternion::from_euler(0.0, std::f32::consts::FRAC_PI_2, 0.0);
        println!("{q}");
        assert_ulps_eq!(q.apply(input), Vec3::unit_y());
    }
    #[test]
    fn rotate_y_90() {
        let mut q = Quaternion::identity();
        // Rotate by 90 degrees
        q.rotate_y(std::f32::consts::FRAC_PI_2);
        println!("Q is now {q}");
        assert_ulps_eq!(
            q,
            Quaternion::new(2.0.sqrt() / 2.0, 0.0, 2.0.sqrt() / 2.0, 0.0)
        );
    }
    #[test]
    fn rotate_z_90() {
        let mut q = Quaternion::identity();
        // Rotate by 90 degrees
        q.rotate_z(std::f32::consts::FRAC_PI_2);
        assert_ulps_eq!(
            q,
            Quaternion::new(2.0.sqrt() / 2.0, 0.0, 0.0, 2.0.sqrt() / 2.0)
        );
    }
}
