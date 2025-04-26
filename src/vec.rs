use std::ops;

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
pub struct Vec3(pub [f32; 3]);

impl Default for Vec3 {
    fn default() -> Self {
        Self::zero()
    }
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3([x, y, z])
    }

    pub fn all(v: f32) -> Vec3 {
        Vec3([v, v, v])
    }

    pub fn zero() -> Vec3 {
        Vec3([0., 0., 0.])
    }

    pub fn unit_x() -> Vec3 {
        Vec3::new(1.0, 0.0, 0.0)
    }

    pub fn unit_y() -> Vec3 {
        Vec3::new(0.0, 1.0, 0.0)
    }

    pub fn unit_z() -> Vec3 {
        Vec3::new(0.0, 0.0, 1.0)
    }

    #[inline(always)]
    pub fn x(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn y(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn z(&self) -> f32 {
        self.0[2]
    }

    /// Mutable reference to x
    pub fn mx(&mut self) -> &mut f32 {
        &mut self.0[0]
    }

    /// Mutable reference to y
    pub fn my(&mut self) -> &mut f32 {
        &mut self.0[1]
    }

    /// Mutable reference to z
    pub fn mz(&mut self) -> &mut f32 {
        &mut self.0[2]
    }

    pub fn set_x(&mut self, x: f32) {
        self.0[0] = x;
    }

    pub fn set_y(&mut self, y: f32) {
        self.0[1] = y;
    }

    pub fn set_z(&mut self, z: f32) {
        self.0[2] = z;
    }

    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(&self) -> f32 {
        self.dot(*self)
    }

    pub fn dot(&self, rhs: Vec3) -> f32 {
        self.x() * rhs.x() + self.y() * rhs.y() + self.z() * rhs.z()
    }

    pub fn cross(&self, rhs: Vec3) -> Vec3 {
        Vec3([
            self.y() * rhs.z() - self.z() * rhs.y(),
            self.z() * rhs.x() - self.x() * rhs.z(),
            self.x() * rhs.y() - self.y() * rhs.x(),
        ])
    }

    pub fn normalized(self) -> Vec3 {
        self * self.length().recip()
    }

    pub fn min_extrema(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            f32::min(self.x(), v.x()),
            f32::min(self.y(), v.y()),
            f32::min(self.z(), v.z()),
        )
    }

    pub fn min_extrema_3(&self, v1: Vec3, v2: Vec3) -> Vec3 {
        Vec3::new(
            f32::min(f32::min(self.x(), v1.x()), v2.x()),
            f32::min(f32::min(self.y(), v1.y()), v2.y()),
            f32::min(f32::min(self.z(), v1.z()), v2.z()),
        )
    }

    pub fn min_extrema_4(&self, v1: Vec3, v2: Vec3, v3: Vec3) -> Vec3 {
        Vec3::new(
            f32::min(f32::min(self.x(), v1.x()), f32::min(v2.x(), v3.x())),
            f32::min(f32::min(self.y(), v1.y()), f32::min(v2.y(), v3.y())),
            f32::min(f32::min(self.z(), v1.z()), f32::min(v2.z(), v3.z())),
        )
    }

    pub fn max_extrema(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            f32::max(self.x(), v.x()),
            f32::max(self.y(), v.y()),
            f32::max(self.z(), v.z()),
        )
    }

    pub fn max_extrema_3(&self, v1: Vec3, v2: Vec3) -> Vec3 {
        Vec3::new(
            f32::max(f32::max(self.x(), v1.x()), v2.x()),
            f32::max(f32::max(self.y(), v1.y()), v2.y()),
            f32::max(f32::max(self.z(), v1.z()), v2.z()),
        )
    }

    pub fn max_extrema_4(&self, v1: Vec3, v2: Vec3, v3: Vec3) -> Vec3 {
        Vec3::new(
            f32::max(f32::max(self.x(), v1.x()), f32::max(v2.x(), v3.x())),
            f32::max(f32::max(self.y(), v1.y()), f32::max(v2.y(), v3.y())),
            f32::max(f32::max(self.z(), v1.z()), f32::max(v2.z(), v3.z())),
        )
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

impl_binary_op!(Add : add => (lhs: Vec3, rhs: Vec3) -> Vec3 {
    Vec3([
        lhs.x() + rhs.x(),
        lhs.y() + rhs.y(),
        lhs.z() + rhs.z(),
    ])
});

impl_binary_op!(Sub : sub => (lhs: Vec3, rhs: Vec3) -> Vec3 {
    Vec3([
        lhs.x() - rhs.x(),
        lhs.y() - rhs.y(),
        lhs.z() - rhs.z(),
    ])
});

impl_binary_op!(Mul : mul => (lhs: Vec3, rhs: f32) -> Vec3 {
    Vec3([
        lhs.x() * rhs,
        lhs.y() * rhs,
        lhs.z() * rhs,
    ])
});

impl_binary_op!(Mul : mul => (lhs: f32, rhs: Vec3) -> Vec3 {
    Vec3([
        rhs.x() * lhs,
        rhs.y() * lhs,
        rhs.z() * lhs,
    ])
});

impl_binary_op!(Div : div => (lhs: Vec3, rhs: f32) -> Vec3 {
    Vec3([
        lhs.x() / rhs,
        lhs.y() / rhs,
        lhs.z() / rhs,
    ])
});

impl ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl ops::SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ops::MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl ops::DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Self::Output {
        Vec3([-self.x(), -self.y(), -self.z()])
    }
}

impl Into<cgmath::Vector3<f32>> for Vec3 {
    fn into(self) -> cgmath::Vector3<f32> {
        cgmath::Vector3::<f32> {
            x: self.x(),
            y: self.y(),
            z: self.z(),
        }
    }
}

impl From<cgmath::Vector3<f32>> for Vec3 {
    fn from(v: cgmath::Vector3<f32>) -> Self {
        Self([v.x, v.y, v.z])
    }
}

impl From<cgmath::Vector4<f32>> for Vec3 {
    fn from(v: cgmath::Vector4<f32>) -> Self {
        Self([v.x, v.y, v.z])
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x(), self.y(), self.z())
    }
}

impl approx::AbsDiffEq for Vec3 {
    //? Not sure why we need to add `as ...`
    type Epsilon = <f32 as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f32::abs_diff_eq(&self.x(), &other.x(), epsilon)
            && f32::abs_diff_eq(&self.y(), &other.y(), epsilon)
            && f32::abs_diff_eq(&self.z(), &other.z(), epsilon)
    }
}

impl approx::RelativeEq for Vec3 {
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
        f32::relative_eq(&self.x(), &other.x(), epsilon, max_relative)
            && f32::relative_eq(&self.y(), &other.y(), epsilon, max_relative)
            && f32::relative_eq(&self.z(), &other.z(), epsilon, max_relative)
    }
}

impl approx::UlpsEq for Vec3 {
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
        f32::ulps_eq(&self.x(), &other.x(), epsilon, max_ulps)
            && f32::ulps_eq(&self.y(), &other.y(), epsilon, max_ulps)
            && f32::ulps_eq(&self.z(), &other.z(), epsilon, max_ulps)
    }
}
