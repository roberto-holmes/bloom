__import structs;

[[vk::binding(3)]]
ConstantBuffer<UBO> ubo;
[[vk::binding(4)]]
StructuredBuffer<Buffers> scene;

[shader("intersection")]
void main() {
	let o = scene[InstanceID()];

	uint object_type = *(uint*)o.primitive;	 // The first word of a primitive should be its object type

	if (object_type == ObjectType::Sphere) {
		let s = (Sphere*)o.primitive;
		let t = hitSphere(s);
		if (t > 0.0) {
			ReportHit(t, object_type, nullptr);
		}
	} else {
		// Hit an unimplemented object
		ReportHit(1000.0, object_type, nullptr);
	}
}

// Ray-Sphere intersection
float hitSphere(const Sphere* sphere) {
	let v = WorldRayOrigin() - sphere.center;
	let a = dot(WorldRayDirection(), WorldRayDirection());
	let b = dot(v, WorldRayDirection());
	let c = dot(v, v) - sphere.radius * sphere.radius;

	// Find roots for the quadratic
	let d = b * b - a * c;

	// If no roots are found, the ray does not intersect with the sphere
	if (d < 0.0) {
		return -1.0;
	}

	// If there is a real solution, find the time at which it takes place
	let sqrt_d = sqrt(d);
	let recip_a = 1. / a;
	let mb = -b;
	let t1 = (mb - sqrt_d) * recip_a;
	let t2 = (mb + sqrt_d) * recip_a;
	let t = t1 > 1e-2 ? t1 : t2;
	if (t <= 1e-2) {
		// Check if the solution is for time = 0
		return -1.0;
	} else {
		return t;
	}
}
