use {
    bytemuck::{Pod, Zeroable},
    cgmath::Vector3,
    std::ops,
};

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Vec3([f32; 3]);

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
        self.dot(self)
    }

    pub fn dot(&self, rhs: &Vec3) -> f32 {
        self.x() * rhs.x() + self.y() * rhs.y() + self.z() * rhs.z()
    }

    pub fn cross(&self, rhs: &Vec3) -> Vec3 {
        Vec3([
            self.y() * rhs.z() - self.z() * rhs.y(),
            self.z() * rhs.x() - self.x() * rhs.z(),
            self.x() * rhs.y() - self.y() * rhs.x(),
        ])
    }

    pub fn normalized(self) -> Vec3 {
        self * self.length().recip()
    }

    pub fn min_extrema(&self, v: &Vec3) -> Vec3 {
        Vec3::new(
            f32::min(self.x(), v.x()),
            f32::min(self.y(), v.y()),
            f32::min(self.z(), v.z()),
        )
    }

    pub fn min_extrema_3(&self, v1: &Vec3, v2: &Vec3) -> Vec3 {
        Vec3::new(
            f32::min(f32::min(self.x(), v1.x()), v2.x()),
            f32::min(f32::min(self.y(), v1.y()), v2.y()),
            f32::min(f32::min(self.z(), v1.z()), v2.z()),
        )
    }

    pub fn min_extrema_4(&self, v1: &Vec3, v2: &Vec3, v3: &Vec3) -> Vec3 {
        Vec3::new(
            f32::min(f32::min(self.x(), v1.x()), f32::min(v2.x(), v3.x())),
            f32::min(f32::min(self.y(), v1.y()), f32::min(v2.y(), v3.y())),
            f32::min(f32::min(self.z(), v1.z()), f32::min(v2.z(), v3.z())),
        )
    }

    pub fn max_extrema(&self, v: &Vec3) -> Vec3 {
        Vec3::new(
            f32::max(self.x(), v.x()),
            f32::max(self.y(), v.y()),
            f32::max(self.z(), v.z()),
        )
    }

    pub fn max_extrema_3(&self, v1: &Vec3, v2: &Vec3) -> Vec3 {
        Vec3::new(
            f32::max(f32::max(self.x(), v1.x()), v2.x()),
            f32::max(f32::max(self.y(), v1.y()), v2.y()),
            f32::max(f32::max(self.z(), v1.z()), v2.z()),
        )
    }

    pub fn max_extrema_4(&self, v1: &Vec3, v2: &Vec3, v3: &Vec3) -> Vec3 {
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

impl Into<Vector3<f32>> for Vec3 {
    fn into(self) -> Vector3<f32> {
        Vector3::<f32> {
            x: self.x(),
            y: self.y(),
            z: self.z(),
        }
    }
}
