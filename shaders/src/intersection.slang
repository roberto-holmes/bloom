__import structs;
__import helpers;
__import constants;

// (Binding, Set)
[[vk::binding(3, 0)]]
ConstantBuffer<UBO> ubo;
[[vk::binding(4, 0)]]
StructuredBuffer<Buffers> scene;
[[vk::binding(6, 0)]]
RWTexture2D<float4> ocean_map;

[shader("intersection")]
void main() {
	let o = scene[InstanceID()];
	CustomAttributes att;
	att.hit_type = 0;

	let object_type = (uint) * (uint64_t*)o.primitive;	// The first word of a primitive should be its object type

	if (object_type == ObjectType::Sphere) {
		let s = (Sphere*)o.primitive;

		let center = mul(ObjectToWorld(), float4(0.0, 0.0, 0.0, 1.0));
		let t = hit_sphere(center, s.radius);
		if (t < F32_MAX) {
			ReportHit(t, object_type, att);
		}
	} else if (object_type == ObjectType::Ocean) {
		let o = (Ocean*)o.primitive;
		let p = hit_ocean(o);
		if (p.r < F32_MAX) {
			// TODO: Figure out how best to give the rest of the data to the hit shader
			ReportHit(p.r, object_type, att);
		}
	} else if (object_type == ObjectType::Lentil) {
		let l = (Lentil*)o.primitive;
		uint hit_type = 0;
		let t = hit_lentil(l, hit_type);
		if (t < F32_MAX) {
			// printf("%u\n", att.hit_type);
			att.hit_type = hit_type;
			ReportHit(t, object_type + hit_type, att);
		}
	} else {
		// Hit an unimplemented object
		ReportHit(1000.0, object_type, att);
	}
}

float hit_aabb(const AABB aabb) {
	float3 invDir = 1.0 / WorldRayDirection();
	float3 tbot = invDir * (aabb.min - WorldRayOrigin());
	float3 ttop = invDir * (aabb.max - WorldRayOrigin());
	float3 tmin = min(ttop, tbot);
	float3 tmax = max(ttop, tbot);
	float t0 = max(tmin.x, max(tmin.y, tmin.z));
	float t1 = min(tmax.x, min(tmax.y, tmax.z));
	return t1 > max(t0, 0.0) ? t0 : F32_MAX;
}

// Ray-Sphere intersection
float hit_sphere(float3 center, float radius) {
	let v = WorldRayOrigin() - center;
	let a = dot(WorldRayDirection(), WorldRayDirection());
	let b = dot(v, WorldRayDirection());
	let c = dot(v, v) - radius * radius;

	// Find roots for the quadratic
	let d = b * b - a * c;

	// If no roots are found, the ray does not intersect with the sphere
	if (d < 0.0) {
		return F32_MAX;
	}

	// If there is a real solution, find the time at which it takes place
	let sqrt_d = sqrt(d);
	let recip_a = 1. / a;
	let mb = -b;
	let t1 = (mb - sqrt_d) * recip_a;
	let t2 = (mb + sqrt_d) * recip_a;
	let t = t1 > EPSILON ? t1 : t2;
	if (t <= EPSILON) {
		// Check if the solution is for time = 0
		return F32_MAX;
	} else {
		return t;
	}
}

// Ray-Sphere intersection
void hit_sphere_2(float3 center, float radius, out float t1, out float t2) {
	let v = WorldRayOrigin() - center;
	let a = dot(WorldRayDirection(), WorldRayDirection());
	let b = dot(v, WorldRayDirection());
	let c = dot(v, v) - radius * radius;

	// Find roots for the quadratic
	let d = b * b - a * c;

	// If no roots are found, the ray does not intersect with the sphere
	if (d < 0.0) {
		t1 = F32_MAX;
		t2 = F32_MAX;
	} else {
		// If there is a real solution, find the time at which it takes place
		let sqrt_d = sqrt(d);
		let recip_a = 1. / a;
		let mb = -b;
		t1 = (mb - sqrt_d) * recip_a;
		t2 = (mb + sqrt_d) * recip_a;
	}
}

float hit_cylinder(float3 center, float3 axis, float radius, float end_1, float end_2) {
	let a = dot(cross(WorldRayDirection(), axis), cross(WorldRayDirection(), axis));
	let b = dot(cross(WorldRayDirection(), axis), cross(WorldRayOrigin() - center, axis));
	let c = dot(cross(WorldRayOrigin() - center, axis), cross(WorldRayOrigin() - center, axis)) - radius * radius;

	// Find roots for the quadratic
	let d = b * b - a * c;

	// If no roots are found, the ray does not intersect with the cylinder
	if (d < 0.0) {
		return F32_MAX;
	}

	// If there is a real solution, find the time at which it takes place
	let sqrt_d = sqrt(d);
	let recip_a = 1. / a;
	let mb = -b;
	let t1 = (mb - sqrt_d) * recip_a;
	let t2 = (mb + sqrt_d) * recip_a;

	let pos1 = point_on_ray(t1) - center;
	let pos2 = point_on_ray(t2) - center;

	let axial_pos_1 = abs(dot(pos1, axis));
	let axial_pos_2 = abs(dot(pos2, axis));

	if (t1 > EPSILON && axial_pos_1 >= end_1 && axial_pos_1 <= end_2) {
		return t1;
	} else if (t2 > EPSILON && axial_pos_2 >= end_1 && axial_pos_2 <= end_2) {
		return t2;
	} else {
		return F32_MAX;
	}
}

float hit_plane(float3 pos, float3 normal) {
	let D = dot(pos, normal);
	let den = dot(normal, WorldRayDirection());
	if (den == 0.0) {
		// Parallel ray so no hit
		return F32_MAX;
	}
	return (D - dot(normal, WorldRayOrigin())) / den;
}

float hit_spherical(float3 pos, float3 axis, float sphere_radius, float edge_offset) {
	float t_sphere_1, t_sphere_2;
	hit_sphere_2(pos - sphere_radius * axis, sphere_radius, t_sphere_1, t_sphere_2);

	let sphere_hit_location_1 = point_on_ray(t_sphere_1);
	let sphere_hit_location_2 = point_on_ray(t_sphere_2);

	if (t_sphere_1 < F32_MAX && edge_offset > 0 && dot(sphere_hit_location_1 - pos, axis) < edge_offset) {
		return t_sphere_1;
	} else if (t_sphere_1 < F32_MAX && edge_offset < 0 && dot(sphere_hit_location_1 - pos, axis) > edge_offset) {
		return t_sphere_1;
	} else if (t_sphere_2 < F32_MAX && edge_offset > 0 && dot(sphere_hit_location_2 - pos, axis) < edge_offset) {
		return t_sphere_2;
	} else if (t_sphere_2 < F32_MAX && edge_offset < 0 && dot(sphere_hit_location_2 - pos, axis) > edge_offset) {
		return t_sphere_2;
	} else {
		return F32_MAX;
	}
}

float hit_aspherical(float3 pos, float3 axis, float sphere_radius, float conic_constant, float edge_offset) {
	float t_sphere_1, t_sphere_2;
	// get initial values via spherical approximation
	hit_sphere_2(pos - sphere_radius * axis, sphere_radius, t_sphere_1, t_sphere_2);
	if (t_sphere_1 == F32_MAX) {
		t_sphere_1 = 0;	 // if no spherical init value found, start with 0
	}
	if (t_sphere_2 == F32_MAX) {
		t_sphere_2 = 1;	 // if no spherical init value found, start with 1
	}
	newton_raphson(10, t_sphere_1, axis, pos, sphere_radius, conic_constant);
	newton_raphson(10, t_sphere_2, axis, pos, sphere_radius, conic_constant);

	let sphere_hit_location_1 = point_on_ray(t_sphere_1);
	let sphere_hit_location_2 = point_on_ray(t_sphere_2);

	if (t_sphere_1 < F32_MAX && edge_offset > 0 && dot(sphere_hit_location_1 - pos, axis) < edge_offset) {
		return t_sphere_1;
	} else if (t_sphere_1 < F32_MAX && edge_offset < 0 && dot(sphere_hit_location_1 - pos, axis) > edge_offset) {
		return t_sphere_1;
	} else if (t_sphere_2 < F32_MAX && edge_offset > 0 && dot(sphere_hit_location_2 - pos, axis) < edge_offset) {
		return t_sphere_2;
	} else if (t_sphere_2 < F32_MAX && edge_offset < 0 && dot(sphere_hit_location_2 - pos, axis) > edge_offset) {
		return t_sphere_2;
	} else {
		return F32_MAX;
	}
}

float hit_circle(float3 center, float3 normal, float radius) {
	let t = hit_plane(center, normal);
	let hit = point_on_ray(t);

	// Figure out if the hit point is within the circle
	float3 v = hit - center;
	if (dot(v, v) > radius * radius) {
		// Hit was outside
		return F32_MAX;
	}
	return t;
}

float hit_lentil(const Lentil* lentil, out uint hit_type) {
	float3 center = mul(ObjectToWorld(), float4(0.0, 0.0, 0.0, 1.0));
	float3 axis = normalize(mul(ObjectToWorld(), float4(1.0, 0.0, 0.0, 0.0)));
	bool enable_asphere = true;
	float edge_offset_a = aspheric_surface(lentil.radius, lentil.curvature_a, enable_asphere ? lentil.kappa_a : 0);
	float edge_offset_b = aspheric_surface(lentil.radius, lentil.curvature_b, enable_asphere ? lentil.kappa_b : 0);

	// Check for an intersection on the surface of the cyclinder
	let t_cyl = hit_cylinder(center, axis, lentil.radius, -lentil.length - edge_offset_a, lentil.length + edge_offset_b);

	// Come up with guesses for each intersection
	float t_a, t_b;
	if (!enable_asphere || lentil.kappa_a == 0) {
		t_a = hit_spherical(center + axis * lentil.length, axis, lentil.curvature_a, edge_offset_a);
	} else {
		t_a = hit_aspherical(center + axis * lentil.length, axis, lentil.curvature_a, lentil.kappa_a, edge_offset_a);
	}
	if (!enable_asphere || lentil.kappa_b == 0) {
		t_b = hit_spherical(center - axis * lentil.length, -axis, lentil.curvature_b, edge_offset_b);
	} else {
		t_b = hit_aspherical(center - axis * lentil.length, -axis, lentil.curvature_b, lentil.kappa_b, edge_offset_b);
	}
	// t_b = hit_circle(center - axis * lentil.length, -axis, lentil.radius);

	// float t_b = hit_circle(center - axis * lentil.length, axis, lentil.radius);
	// float t_a = hit_plane(axis * lentil.length, axis);
	// float t_b = hit_plane(-axis * lentil.length, axis);

	// If we didn't find an intersection (parallel ray) guess 1
	// t_a = t_a == F32_MAX ? t_a : 1.0;
	// t_b = t_b == F32_MAX ? t_b : 1.0;

	// Check for intersections on each lense surface
	// if (t_a != F32_MAX) {
	// 	newton_raphson(10, t_a, axis, center + axis * lentil.length, lentil.curvature_a, lentil.kappa_a);
	// }
	// if (t_b != F32_MAX) {
	// 	newton_raphson(10, t_b, -axis, center - axis * lentil.length, lentil.curvature_b, lentil.kappa_b);
	// }

	// t_a = isnan(t_a) ? F32_MAX : t_a;
	// t_b = isnan(t_b) ? F32_MAX : t_b;

	// if (ubo.ray_frame_num == 1) {
	// 	printf("%.2f %.2f %.2f\n", t_cyl == F32_MAX ? 0.0 : t_cyl, t_a == F32_MAX ? 0.0 : t_a, t_b == F32_MAX ? 0.0 : t_b);
	// }

	if (t_cyl < F32_MAX && t_cyl < t_a && t_cyl < t_b) {
		hit_type = 1;
		return t_cyl;
	} else if (t_a < F32_MAX && t_a < t_b && t_a < t_cyl) {
		hit_type = 2;
		return t_a;
	} else if (t_b < F32_MAX && t_b < t_a && t_b < t_cyl) {
		hit_type = 3;
		return t_b;
	} else {
		hit_type = 0;
		return F32_MAX;
	}
}

void newton_raphson(uint iterations, inout float t, float3 axis, float3 center, float R, float k) {
	float last_t;
	for (uint i = 0; i < iterations; i++) {
		last_t = t;
		t -= f(t, axis, center, R, k) / f_prime(t, axis, center, R, k);
	}
	if (abs(t - last_t) > 1E-4) {
		t = F32_MAX;
	}
}

float aspheric_surface(float r, float R, float k) {
	let base = r * r / (R * (1 + sqrt(1 - (1 + k) * r * r / (R * R))));
	return -base;
}

float aspheric_surface_prime(float r, float R, float k) {
	let num = R * r * sqrt(1 - ((1 + k) * r * r) / (R * R));
	let den = R * R - (1 + k) * r * r;
	return -num / den;
}

void r_abc(float t, float3 axis, float3 center, out float a, out float b, out float c) {
	a = dot(cross(WorldRayDirection(), axis), cross(WorldRayDirection(), axis));
	b = 2 * dot(cross(WorldRayDirection(), axis), cross(WorldRayOrigin() - center, axis));
	c = dot(cross(WorldRayOrigin() - center, axis), cross(WorldRayOrigin() - center, axis));
}

float r(float t, float a, float b, float c) {
	return sqrt(t * t * a + t * b + c);
}

float r_prime(float t, float a, float b, float c) {
	return (2 * a * t + b) / (2 - sqrt(t * (a * t + b) + c));
}

float f(float t, float3 axis, float3 center, float R, float k) {
	float a;
	float b;
	float c;
	r_abc(t, axis, center, a, b, c);
	return aspheric_surface(r(t, a, b, c), R, k) - dot(WorldRayOrigin() + t * WorldRayDirection() - center, axis);
}

float f_prime(float t, float3 axis, float3 center, float R, float k) {
	float a;
	float b;
	float c;
	r_abc(t, axis, center, a, b, c);

	// Apply chain rule
	return aspheric_surface_prime(r(t, a, b, c), R, k) * r_prime(t, a, b, c) - dot(WorldRayDirection(), axis);
}

float hit_ocean(const Ocean* ocean) {
	// Get initial t from intersection with AABB
	// TODO: Get AABB size from Ocean object
	let aabb = AABB(float3(-10.0, -1.0, -10.0), float3(10.0, 1.0, 10.0));
	let aabb_hit = hit_aabb(aabb);

	// TODO: Tweak values to improve performance
	float t_max = 20.0;
	float dt = 0.01;
	float lh = 0.0;
	float ly = 0.0;

	// Ray march to see if we hit the ocean
	for (float t = aabb_hit; t < aabb_hit + t_max; t += dt) {
		let p = WorldRayOrigin() + WorldRayDirection() * t;
		var point = get_ocean_point(p.xz * 100.0, ocean.size, ocean_map);
		// float point = 0.0;
		if (!isfinite(point)) {
			// There is a discontinuity in the ocean
			point.r = 1000.0;
		}
		if (p.y < point) {
			// The ray marched point is below the surface so we consider it a hit
			// interpolate to get the intersection
			return t - dt + dt * (lh - ly) / (p.y - ly - point + lh);
		}
		// Accuracy goes down as we are checking further away
		// TODO: Figure out current total ray length (including bounces) and use this as accuracy should also be lower for subsequent bounces
		if (t > 0) {
			dt = 0.05 * t;
		} else {
			dt *= 1.001;
		}
		lh = point;
		ly = p.y;
	}
	return float(F32_MAX);
}
