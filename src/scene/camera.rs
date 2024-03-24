use nalgebra::{Matrix3, Point3};

use crate::{pipeline::ShaderData, util::as_u8_slice};


// #[repr(C, align(16))]
// struct ShaderVec3([u8; 48]);

// #[repr(C, align(16))]
// struct ShaderMatrix4x3([u8; 48]);

// impl ShaderData for ShaderMatrix4x3 {

// }


#[repr(C)] 
struct CameraData {
    col0: [f32; 3],
    _pad0: f32,
    col1: [f32; 3],
    _pad1: f32,
    col2: [f32; 3],
    _pad2: f32,
    position: [f32; 3],
    _pad3: f32,
}

impl CameraData {
    fn new(transform: Matrix3<f32>, position: Point3<f32>) -> Self {
        
        Self {  
            col0: [transform[(0, 0)],  transform[(1, 0)],  transform[(2, 0)]],
            col1: [transform[(0, 1)],  transform[(1, 1)],  transform[(2, 1)]],
            col2: [transform[(0, 2)],  transform[(1, 2)],  transform[(2, 2)]],
            position: [position.x, position.y, position.z],

            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        }
    }
}

impl ShaderData for CameraData {
    fn as_u8_slice<'a>(&'a self) -> &'a [u8] {
        unsafe { as_u8_slice(self) }
    }
}

pub struct Camera {
    position: Point3<f32>,
    transform: Matrix3<f32>,
}

impl Camera {
    pub fn new(
        position: Point3<f32>,
        transform: Matrix3<f32>,
    ) -> Self {
        Self {
            position,
            transform,
        }
    }

    pub fn serialize(&self) -> impl ShaderData {
        CameraData::new(self.transform, self.position)
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Point3::origin(),
            transform: Matrix3::new(
                1.0, 0.0, -0.5,
                0.0, 1.0, -0.5,
                0.0, 0.0, 1.0,
            ),
        }
    }
}


