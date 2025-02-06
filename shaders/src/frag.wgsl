// ----------------------- RNG Tools ----------------------- 
struct Rng {
  state: u32,
};
var<private> rng: Rng;

fn init_rng(pixel: vec2u, width: u32, frame_num: u32) {
    // Seed the PRNG using the scalar index of the pixel and the current frame count.
    let seed = (pixel.x + pixel.y * width) ^ jenkins_hash(frame_num);
    rng.state = jenkins_hash(seed);
}

// A slightly modified version of the "One-at-a-Time Hash" function by Bob Jenkins.
// See https://www.burtleburtle.net/bob/hash/doobs.html
fn jenkins_hash(i: u32) -> u32 {
    var x = i;
    x += x << 10u;
    x ^= x >> 6u;
    x += x << 3u;
    x ^= x >> 11u;
    x += x << 15u;
    return x;
}

// The 32-bit "xor" function from Marsaglia G., "Xorshift RNGs", Section 3.
fn xorshift32() -> u32 {
    var x = rng.state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng.state = x;
    return x;
}

// Returns a random float in the range [0...1]. This sets the floating point exponent to zero and
// sets the most significant 23 bits of a random 32-bit unsigned integer as the mantissa. That
// generates a number in the range [1, 1.9999999], which is then mapped to [0, 0.9999999] by
// subtraction. See Ray Tracing Gems II, Section 14.3.4.
fn rand_f32() -> f32 {
    return bitcast<f32>(0x3f800000u | (xorshift32() >> 9u)) - 1.;
}

// ----------------------- Fragment shader ----------------------- 
// Order of members in structs is very important for aligning purposes
struct CameraUniforms {
    origin: vec3f,
    focal_distance: f32,
    u: vec3f,
    vfov: f32,
    v: vec3f,
    dof_scale: f32,
    w: vec3f,
}

struct Uniforms {
    camera: CameraUniforms,
    frame_num: u32,
    width: u32,
    height: u32,
};

struct Material {
    albedo: vec3f,
    alpha: f32,
    refraction_index: f32,
    smoothness: f32,
    emissivity: f32,
    emission_strength: f32,
    emission_colour: vec3f,
}

struct Sphere {
    center: vec3f,
    radius: f32,
    material: u32,
    is_selected: u32,
}

struct Quad {
    q: vec3f,
    is_selected: u32,
    u: vec3f,
    v: vec3f,
    normal: vec3f,
    d: f32,
    w: vec3f,
    material: u32,
}

struct Triangle {
    a: vec3f,
    is_selected: u32,
    b: vec3f,
    material: u32,
    c: vec3f,
}

struct AABB {
    min: vec3f,
    left_child_index: u32,
    max: vec3f,
    right_child_index: u32,
    object_type: u32,
    object_index: u32,
    is_populated: u32,
}

// @group(0) @binding(0) var radiance_samples_old: texture_2d<f32>;
// @group(0) @binding(1) var radiance_samples_new: texture_storage_2d<rgba32float, write>;
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// @group(0) @binding(3) var<uniform> materials: array<Material, 10>;
// @group(0) @binding(4) var<storage, read> bvh: array<AABB>;
// @group(0) @binding(5) var<storage, read> spheres: array<Sphere>;
// @group(0) @binding(6) var<storage, read> quads: array<Quad>;
// @group(0) @binding(7) var<storage, read> triangles: array<Triangle>;


struct Ray {
    origin: vec3f,
    direction: vec3f,
}

struct Intersection {
    normal: vec3f,
    t: f32,
    material: Material,
}

struct Scatter {
    attenuation: vec3f,
    ray: Ray,
}

const TAU: f32 = 6.283185307;
const F32_MAX: f32 = 3.40282346638528859812e+38;
const U32_MAX: u32 = 4294967295;
const EPSILON: f32 = 1e-2;

const QUAD_SELECT_WIDTH: f32 = 0.01;

const OBJECT_TYPE_SPHERE: u32 = 0;
const OBJECT_TYPE_QUAD: u32 = 1;
const OBJECT_TYPE_TRIANGLE: u32 = 2;

const MAX_PATH_LENGTH: u32 = 8u;

fn sky_colour(ray: Ray) -> vec3f {
    // Get a value that goes from 1 to 0 as you go down
    let t = 0.5 * (normalize(ray.direction).y + 1.);
    // Make a vertical linear gradient from light blue to white
    return (1. - t) * vec3(1.) + t * vec3(0.3, 0.5, 1.);
    // or use a rough approximation of twilight (From light red to white)
    // return (1. - t) * vec3(1.) + t * vec3(1., 0.5, 0.3);
    // return vec3(0.);
}

// Get the position of point on a ray at a given time
fn point_on_ray(ray: Ray, t: f32) -> vec3f {
    return ray.origin + t * ray.direction;
}

fn generate_random_unit_vector() -> vec3f{
    return normalize(vec3f(rand_f32()*2.-1., rand_f32()*2.-1., rand_f32()*2.-1.));
}

// fn is_reflective_schlick(cosine: f32, refraction_index: f32) -> bool {
//     var r0 = (1. - refraction_index) / (1. + refraction_index);
//     r0 = r0 * r0;
//     return (r0 + (1. - r0) * pow((1. - cosine), 5.)) > rand_f32();
// }

// fn reflect_ray(input_ray: Ray, hit: Intersection) -> Scatter {
//     let lambertian_reflection = hit.normal + generate_random_unit_vector();
//     let metallic_reflection = reflect(input_ray.direction, hit.normal);
//     let reflected = mix(lambertian_reflection, metallic_reflection, hit.material.smoothness);
//     // Bump the start of the reflected ray a little bit off the surface to
//     // try to minimize self intersections due to floating point errors
//     let output_ray = Ray(point_on_ray(input_ray, hit.t) + hit.normal * EPSILON, reflected);
//     let attenuation = hit.material.albedo;
//     return Scatter(attenuation, output_ray);
// }

// fn dielectric_scatter(input_ray: Ray, hit: Intersection) -> Scatter {
//     // Figure out which side of the surface we are hitting
//     let normal = faceForward(hit.normal, input_ray.direction, hit.normal);
//     let refraction_index = select(hit.material.refraction_index, 1. / hit.material.refraction_index, dot(input_ray.direction, hit.normal) > 0.);

//     let input_direction = normalize(input_ray.direction);
//     var output_ray_direction = refract(input_direction, normal, refraction_index);

//     let cos_theta = min(dot(-input_direction, normal), 1.0);

//     var output_ray = input_ray;
//     // If angle is less than the critical angle, reflection occurs instead and the function returns vec3(0.)
//     if (output_ray_direction.x == 0. && output_ray_direction.y == 0. && output_ray_direction.z == 0.) || is_reflective_schlick(cos_theta, hit.material.refraction_index) {
//         output_ray_direction = reflect(input_direction, normal);
//         output_ray = Ray(point_on_ray(input_ray, hit.t) + normal * EPSILON, output_ray_direction);
//     } else {
//         output_ray = Ray(point_on_ray(input_ray, hit.t), output_ray_direction);
//     }

//     let attenuation = hit.material.albedo;
//     return Scatter(attenuation, output_ray);
// }

// fn scatter(input_ray: Ray, hit: Intersection) -> Scatter {
//     // Probability of refracting
//     if hit.material.alpha < rand_f32() {
//         return dielectric_scatter(input_ray, hit);
//     }
//     return reflect_ray(input_ray, hit);
// }

// // Create an empty intersection
// fn no_intersection() -> Intersection {
//     return Intersection(vec3(0.), -1., Material(vec3f(0.), 0., 0., 0., 0., 0., vec3f(0.)));
// }

// // Calculate if an intersection has occured
// fn is_intersection(hit: Intersection) -> bool {
//     return hit.t > 0.;
// }

// fn is_light(hit: Intersection) -> bool {
//     return hit.material.emissivity > rand_f32();
// }

// fn intersect_sphere(ray: Ray, sphere: Sphere) -> Intersection {
//     let v = ray.origin - sphere.center;
//     let a = dot(ray.direction, ray.direction);
//     let b = dot(v, ray.direction);
//     let c = dot(v, v) - sphere.radius * sphere.radius;

//     // Find roots for the quadratic
//     let d = b * b - a * c;

//     // If no roots are found, the ray does not intersect with the sphere
//     if d < 0. {
//         return no_intersection();
//     }

//     // If there is a real solution, find the time at which it takes place
//     let sqrt_d = sqrt(d);
//     let recip_a = 1. / a;
//     let mb = -b;
//     let t1 = (mb - sqrt_d) * recip_a;
//     let t2 = (mb + sqrt_d) * recip_a;
//     let t = select(t2, t1, t1 > EPSILON);
//     if t <= EPSILON {
//         // Check if the solution is for time = 0
//         return no_intersection();
//     }

//     let p = point_on_ray(ray, t);
//     let N = (p - sphere.center) / sphere.radius;

//     // Highlight edges of selected object
//     if (sphere.is_selected > 0) && (d <= (.05 * sphere.radius)) {
//         return Intersection(N, t, Material(vec3(1., 0., 0.), 1., 0., 0., 1., 1., vec3f(1., 0., 0.)));
//     }

//     return Intersection(N, t, materials[sphere.material]);
// }

// fn intersect_quad(ray: Ray, quad: Quad) -> Intersection {
//     // Check if ray interects the plane
//     let denom = dot(quad.normal, ray.direction);

//     // Ray is parallel to the plane
//     if abs(denom) < 1e-8 {
//         return no_intersection();
//     }

//     let t = (quad.d - dot(quad.normal, ray.origin)) / denom;
//     if t < EPSILON {
//         return no_intersection();
//     }

//     let p = point_on_ray(ray, t);

//     // Check if the intersection is within the quad
//     let planar_to_hit = p - quad.q;
//     let alpha = dot(quad.w, cross(planar_to_hit, quad.v));
//     let beta = dot(quad.w, cross(quad.u, planar_to_hit));

//     if !(0 <= alpha && alpha <= 1 && 0 <= beta && beta <= 1) {
//         return no_intersection();
//     }

//     if quad.is_selected > 0 && (alpha > (1. - QUAD_SELECT_WIDTH) || alpha < QUAD_SELECT_WIDTH || beta > (1. - QUAD_SELECT_WIDTH) || beta < QUAD_SELECT_WIDTH) {
//         return Intersection(quad.normal, t, Material(vec3(1., 0., 0.), 1., 0., 0., 1., 1., vec3f(1., 0., 0.)));
//     }

//     return Intersection(quad.normal, t, materials[quad.material]);
// }

// fn intersect_tri(ray: Ray, triangle: Triangle) -> Intersection {
//     // Möller–Trumbore
//     let e1 = triangle.b - triangle.a;
//     let e2 = triangle.c - triangle.a;

//     let ray_cross_e2 = cross(ray.direction, e2);
//     let det = dot(e1, ray_cross_e2);

//     if abs(det) < EPSILON {
//         return no_intersection(); // Ray is parallel to the triangle
//     }

//     let inv_det = 1.0 / det;
//     let s = ray.origin - triangle.a;
//     let u = inv_det * dot(s, ray_cross_e2);
//     if u < 0.0 || 1.0 < u {
//         return no_intersection();
//     }

//     let s_cross_e1 = cross(s, e1);
//     let v = inv_det * dot(ray.direction, s_cross_e1);
//     if v < 0.0 || u + v > 1.0 {
//         return no_intersection();
//     }
    
// 	// At this stage we can compute t to find out where the intersection point is on the line.
//     let t = inv_det * dot(e2, s_cross_e1);

//     if t > EPSILON { // ray intersection
//         return Intersection(cross(e1, e2), t, materials[triangle.material]);
// 	}

//     return no_intersection();
// }

// fn intersect_aabb(ray: Ray, node: AABB) -> bool {
//     let t_aabb_min = vec3(
//         (node.min.x - ray.origin.x) / ray.direction.x,
//         (node.min.y - ray.origin.y) / ray.direction.y,
//         (node.min.z - ray.origin.z) / ray.direction.z
//     );

//     let t_aabb_max = vec3(
//         (node.max.x - ray.origin.x) / ray.direction.x,
//         (node.max.y - ray.origin.y) / ray.direction.y,
//         (node.max.z - ray.origin.z) / ray.direction.z
//     );

//     // If the ray is coming in the opposite direction, the min and max values
//     // stored by the aabb will be reversed from our perspective
//     let t_min = min(t_aabb_min, t_aabb_max);
//     let t_max = max(t_aabb_min, t_aabb_max);

//     // Compare the biggest t_min and the smallest t_max to see 
//     // if there exists a time when the ray is within all dimensions
//     let max_t_min = max(max(t_min.x, t_min.y), t_min.z);
//     let min_t_max = min(min(t_max.x, t_max.y), t_max.z);

//     if min_t_max < 0. {
//         return false;
//     }

//     return max_t_min < min_t_max;
// }

// fn intersect_scene(ray: Ray) -> Intersection {
//     var node_stack = array<AABB, 16>();
//     var stack_index = 0;

//     node_stack[stack_index] = bvh[0];
//     stack_index += 1;

//     var closest_hit = no_intersection();
//     closest_hit.t = F32_MAX;

//     // while stack_index > 0 {
//     for (var i = 0; i < 100; i++) {
//         stack_index -= 1;
//         let node = node_stack[stack_index];

//         if node.is_populated == 0 {
//             // We have reached the end of the BVH
//             break;
//         }

//         if intersect_aabb(ray, node) {
//             if node.object_index == U32_MAX {
//                 // Not a leaf node
//                 node_stack[stack_index] = bvh[node.left_child_index];
//                 stack_index += 1;
//                 node_stack[stack_index] = bvh[node.right_child_index];
//                 stack_index += 1;
//             } else {
//                 // Leaf node
//                 var hit = no_intersection();
//                 if node.object_type == OBJECT_TYPE_SPHERE {
//                     hit = intersect_sphere(ray, spheres[node.object_index]);
//                 } else if node.object_type == OBJECT_TYPE_QUAD {
//                     hit = intersect_quad(ray, quads[node.object_index]);
//                 } else if node.object_type == OBJECT_TYPE_TRIANGLE {
//                     hit = intersect_tri(ray, triangles[node.object_index]);
//                 }
//                 if hit.t > 0. && hit.t < closest_hit.t {
//                     closest_hit = hit;
//                 }
//             }
//         }
//         if stack_index == 0 {
//             break;
//         }
//     }

//     if closest_hit.t < F32_MAX {
//         return closest_hit;
//     }
//     return no_intersection();
// }

fn get_random_in_unit_disk() -> vec3f {
    let r = rand_f32() * 2. - 1.;
    let theta = rand_f32() * TAU;
    return vec3(r * cos(theta), r * sin(theta), 0.);
}

@fragment
fn main(@builtin(position) pos: vec4f) -> @location(0) vec4<f32> {
    // Seed the Random Number Generator
    init_rng(vec2u(pos.xy), uniforms.width, uniforms.frame_num);

    let aspect_ratio = f32(uniforms.width) / f32(uniforms.height);

    let offset = get_random_in_unit_disk().xy;
    // Add some jitter and normalize the viewport coordinates (0,0 top left; 1,1 bottom right)
    var uv = (pos.xy + offset) / vec2f(f32(uniforms.width-1u), f32(uniforms.height-1u));

    // Map 'uv' from y-down (normalized) viewport coordinates to camera coordinates
    // (y-up, x-right, right hand, screen height is 2 units)
    uv = (2. * uv - vec2(1.)) * vec2(aspect_ratio,  -1.);

    let viewport_scale_factor = 2. * uniforms.camera.focal_distance * tan(uniforms.camera.vfov / 2.);

    uv *= vec2(viewport_scale_factor);

    // Compute the world-space ray direction by rotating the camera-space vector into a new basis
    let camera_rotation = mat3x3(uniforms.camera.u, uniforms.camera.v, uniforms.camera.w);
    let dof_offset = camera_rotation * get_random_in_unit_disk() * uniforms.camera.dof_scale;
    let direction = camera_rotation * vec3(uv, uniforms.camera.focal_distance) - dof_offset;
    let origin = uniforms.camera.origin + dof_offset;
    var ray = Ray(origin, normalize(direction));
    var throughput = vec3f(1.);
    var radiance_sample = vec3(0.);

    // Propagate the ray into the spheres and get the final colours
    var path_length = 0u;
    while path_length < MAX_PATH_LENGTH {
        // let hit = intersect_scene(ray);
        // if !is_intersection(hit) {
            // If not intersection was found, return the colour of the sky and terminate the path
            radiance_sample += throughput * sky_colour(ray);
            break;
        // }
        // if is_light(hit) {
        //     radiance_sample += throughput * hit.material.emission_colour * hit.material.emission_strength;
        // }

        // let scattered = scatter(ray, hit);
        // throughput *= scattered.attenuation;
        // ray = scattered.ray;
        // path_length += 1u;
    }

    // Fetch the old sum of samples
    // var old_sum: vec3f;
    // if uniforms.frame_num > 1 {
    //     old_sum = textureLoad(radiance_samples_old, vec2u(pos.xy), 0).xyz;
    // } else {
    //     old_sum = vec3(0.);
    // }

    // Compute and store the new sum
    // let new_sum = radiance_sample + old_sum;
    // textureStore(radiance_samples_new, vec2u(pos.xy), vec4(new_sum, 0.));

    // Apply gamma correction to go from linear colour space to sRGB (gamma = 2.2)
    // let colour = new_sum / f32(uniforms.frame_num);
    // return vec4(pow(colour, vec3(1. / 2.2)), 1.);
    return vec4(pow(radiance_sample, vec3(1. / 2.2)), 1.);
}
